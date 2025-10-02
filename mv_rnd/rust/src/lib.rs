use ndarray::{Array1, Array2, ArrayView2, ArrayViewMut2, Axis};
use numpy::{PyArray2, PyReadonlyArray2, PyArray1, PyReadonlyArray1};
use pyo3::prelude::*;

/// Compute weighted sums S and Z_agg for block rearrangement.
///
/// This is the hot path - tight loops over large arrays.
/// Rust's zero-cost abstractions and LLVM optimization should give 5-10x over Python.
#[inline(always)]
fn compute_weighted_sums(
    y: &Array2<f64>,
    in_block: &[usize],
    out_block: &[usize],
    tilde_alpha: &Array2<f64>,
    alpha_block: &Array1<f64>,
) -> (Array1<f64>, Array1<f64>) {
    let n = y.nrows();
    let k = tilde_alpha.nrows();
    
    // Compute S = sum of block columns
    let mut s = Array1::<f64>::zeros(n);
    for i in 0..n {
        let mut s_val = 0.0;
        for &j in in_block {
            s_val += y[[i, j]];
        }
        s[i] = s_val;
    }
    
    // Compute Z_agg = weighted sum outside block
    let mut z_agg = Array1::<f64>::zeros(n);
    if !out_block.is_empty() {
        for k_idx in 0..k {
            let alpha_k = alpha_block[k_idx];
            for i in 0..n {
                let mut z_val = 0.0;
                for &j in out_block {
                    z_val += tilde_alpha[[k_idx, j]] * y[[i, j]];
                }
                z_agg[i] += alpha_k * z_val;
            }
        }
    }
    
    (s, z_agg)
}

/// Block rearrangement implemented in pure Rust.
///
/// Args:
///     y: (n, d+K) state matrix (modified in-place)
///     block_mask: (d+K,) boolean mask (as u8)
///     tilde_alpha: (K, d+K) coefficient matrix
#[pyfunction]
fn block_rearrangement_rust<'py>(
    py: Python<'py>,
    y: &PyArray2<f64>,
    block_mask: PyReadonlyArray1<u8>,
    tilde_alpha: PyReadonlyArray2<f64>,
) -> PyResult<()> {
    let block_mask = block_mask.as_array();
    let tilde_alpha_array = tilde_alpha.as_array();
    
    // Find in_block and out_block indices
    let in_block: Vec<usize> = block_mask
        .iter()
        .enumerate()
        .filter(|(_, &v)| v > 0)
        .map(|(i, _)| i)
        .collect();
    
    let out_block: Vec<usize> = block_mask
        .iter()
        .enumerate()
        .filter(|(_, &v)| v == 0)
        .map(|(i, _)| i)
        .collect();
    
    if in_block.is_empty() {
        return Ok(());
    }
    
    // Get mutable view of Y
    let mut y_array = unsafe { y.as_array_mut() };
    let y_readonly = y_array.view();
    
    // Get alpha_block (coefficients for first column in block)
    let alpha_block = tilde_alpha_array.column(in_block[0]).to_owned();
    
    // Compute S and Z_agg
    let (s, z_agg) = compute_weighted_sums(
        &y_readonly.to_owned(),
        &in_block,
        &out_block,
        &tilde_alpha_array.to_owned(),
        &alpha_block,
    );
    
    // Release GIL for sorting (NumPy will be called)
    let s_py = PyArray1::from_array(py, &s);
    let z_agg_py = PyArray1::from_array(py, &z_agg);
    
    // Call NumPy's argsort (it's faster than any pure Rust implementation)
    let np = py.import("numpy")?;
    let sort_z = np.call_method1("argsort", (z_agg_py,))?;
    let sort_s_desc = {
        let sort_s = np.call_method1("argsort", (s_py,))?;
        // Reverse for descending
        let slice = py.eval("lambda x: x[::-1]", None, None)?;
        slice.call1((sort_s,))?
    };
    
    // Convert back to Rust arrays
    let sort_z_arr: PyReadonlyArray1<i64> = sort_z.extract()?;
    let sort_s_arr: PyReadonlyArray1<i64> = sort_s_desc.extract()?;
    
    let sort_z_slice = sort_z_arr.as_slice()?;
    let sort_s_slice = sort_s_arr.as_slice()?;
    
    let n = y_array.nrows();
    
    // Create permutation
    let mut permutation = vec![0usize; n];
    for i in 0..n {
        permutation[sort_z_slice[i] as usize] = sort_s_slice[i] as usize;
    }
    
    // Apply permutation to block columns
    let mut y_block_copy = Array2::<f64>::zeros((n, in_block.len()));
    for (i, row) in y_array.rows().into_iter().enumerate() {
        for (j_idx, &j) in in_block.iter().enumerate() {
            y_block_copy[[i, j_idx]] = row[j];
        }
    }
    
    for i in 0..n {
        let perm_i = permutation[i];
        for (j_idx, &j) in in_block.iter().enumerate() {
            y_array[[i, j]] = y_block_copy[[perm_i, j_idx]];
        }
    }
    
    Ok(())
}

/// Python module
#[pymodule]
fn cbra_core(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(block_rearrangement_rust, m)?)?;
    Ok(())
}
