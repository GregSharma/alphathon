# Alphathon Q1: System Architecture Mermaid Diagrams

## Complete System Data Flow

``` mermaid
graph TB
    subgraph "Layer 1: Data Sources"
        S3T[S3 Parquet<br/>Equity Trades<br/>2024-11-05]
        S3Q[S3 Parquet<br/>Equity Quotes<br/>2024-11-05]
        S3O[S3 Parquet<br/>Option Quotes<br/>downsampled]
        CH[ClickHouse<br/>Kalshi Trades<br/>PRES-2024-KH/DJT]
        JSON[JSON<br/>Fama Weights<br/>factor_weights_tickers.json]
    end
    
    subgraph "Layer 2: CSP Data Ingestion"
        PR1[ParquetReader<br/>Trades]
        PR2[ParquetReader<br/>Quotes]
        PR3[ParquetReader<br/>Options]
        KL[Kalshi Loader<br/>Custom Adapter]
        
        S3T --> PR1
        S3Q --> PR2
        S3O --> PR3
        CH --> KL
    end
    
    subgraph "Layer 3: Stream Demultiplexing"
        DMUX1[Demux by Ticker<br/>40 equity streams]
        DMUX2[Demux by Ticker<br/>40 quote streams]
        DMUX3[Demux by Underlying+Expiry<br/>option chains]
        
        PR1 --> DMUX1
        PR2 --> DMUX2
        PR3 --> DMUX3
    end
    
    subgraph "Layer 4: Feature Engineering"
        LR[Lee-Ready Tagger<br/>directional_indicator<br/>ISO detection]
        AGG[1s Trade Aggregation<br/>15+ features]
        QC[Quote Counter<br/>NBBO updates]
        KF[Kalshi Kalman Filter<br/>Denoised Innovations]
        
        DMUX1 --> LR
        DMUX2 --> QC
        LR --> AGG
        QC --> AGG
        KL --> KF
    end
    
    subgraph "Layer 5: Derived Streams"
        EB[EquityBar1s<br/>iso_flow_intensity<br/>total_slippage<br/>avg_rel_spread]
        KB[KalshiBar1s<br/>filtered_prob<br/>normalized_innovation]
        
        AGG --> EB
        KF --> KB
    end
    
    subgraph "Layer 6: Factor Decomposition"
        FF[Fama Factors Node<br/>Anoush's function]
        BD[Beta Decomposition<br/>Systematic vs Idio]
        
        EB --> FF
        JSON --> FF
        EB --> BD
        FF --> BD
    end
    
    subgraph "Layer 7: Advanced Models"
        VEC[Vectorize Option Chain<br/>quote_list_to_vector]
        RND[RND Extraction<br/>GPR - existing code]
        SUP[RND Superimposition<br/>Kalshi-conditional]
        
        DMUX3 --> VEC
        VEC --> RND
        RND --> SUP
        KB --> SUP
    end
    
    subgraph "Layer 8: Leadership Analysis"
        ML[David's ML Model<br/>Pre-trained Predictor]
        IS[Information Share<br/>David's VECM function]
        
        EB --> ML
        EB --> IS
        BD --> IS
    end
    
    subgraph "Layer 9: Stretch Goals - OPTIONAL"
        CBRA[CBRA Joint Distribution<br/>D+K marginals → joint]
        FFRND[Implied Fama Factor RND<br/>Project joint states]
        
        RND --> CBRA
        CBRA --> FFRND
        FF --> FFRND
    end
    
    subgraph "Layer 10: Outputs & Visualization"
        DASH[Panel Dashboard<br/>4 widgets]
        CSV[CSV Exports<br/>5 files]
        
        KB --> DASH
        EB --> DASH
        FF --> DASH
        IS --> DASH
        RND --> DASH
        
        KB --> CSV
        EB --> CSV
        FF --> CSV
        IS --> CSV
        ML --> CSV
    end
    
    style S3T fill:#e1f5ff
    style S3Q fill:#e1f5ff
    style S3O fill:#e1f5ff
    style CH fill:#ffe1f5
    style JSON fill:#fff4e1
    style EB fill:#d4edda
    style KB fill:#d4edda
    style FF fill:#fff3cd
    style RND fill:#f8d7da
    style IS fill:#cfe2ff
    style DASH fill:#d1ecf1
    style CSV fill:#d1ecf1
```

------------------------------------------------------------------------

## Detailed CSP Node Flow

``` mermaid
graph LR
    subgraph "Per-Ticker Processing"
        T[ts EqTrade]
        Q[ts EqQuote]
        
        T --> TAG{Lee-Ready<br/>Tagger}
        Q -.passive.-> TAG
        
        TAG --> |directional_indicator| BUF[Trade Buffer<br/>1s window]
        Q -.-> |cache mid-price| BUF
        
        BUF --> COMP[Compute Features<br/>ISO/Flow/Slippage]
        COMP --> OUT[ts EquityBar1s]
    end
    
    subgraph "Kalshi Processing"
        KT[ts KalshiTrade]
        KT --> KBUF[1s Buffer]
        KBUF --> KKAL[Kalman Filter<br/>State: p_mean, p_var]
        KKAL --> KOUT[ts KalshiBar1s<br/>normalized_innovation]
    end
    
    subgraph "Factor Processing"
        BASKET{ticker: ts EquityBar1s}
        BASKET --> FNODE[Fama Factors Node<br/>Weighted Portfolio]
        WEIGHTS[factor_weights.json] -.-> FNODE
        FNODE --> FOUT[ts FamaFactors]
    end
    
    subgraph "RND Processing"
        OPT[ts OptQuote for chain]
        OPT --> VEC[quote_list_to_vector<br/>Vectorize]
        VEC --> GPR[RND Extraction<br/>Gaussian Process]
        GPR --> ROUT[ts RNDensity]
    end
```

------------------------------------------------------------------------

## Information Flow Analysis

``` mermaid
graph TD
    subgraph "Macro Event"
        KALSHI[Kalshi Probability Shift<br/>normalized_innovation spike]
    end
    
    subgraph "Market Reactions - Hypothesized Order"
        OPT[Options Market<br/>RND shifts]
        ETF[ETFs<br/>SPY, QQQ react]
        EQ[Individual Equities<br/>AAPL, MSFT lag]
    end
    
    subgraph "Information Channels"
        ISO[ISO Flow<br/>Informed Traders]
        QUOTE[Quote Updates<br/>Market Makers]
    end
    
    subgraph "Factor Decomposition"
        SYS[Systematic Component<br/>Fama Factors]
        IDIO[Idiosyncratic Component<br/>Stock-Specific]
    end
    
    KALSHI --> |shock propagation| OPT
    KALSHI --> |shock propagation| ETF
    KALSHI --> |shock propagation| EQ
    
    OPT --> |leads| ETF
    ETF --> |leads| EQ
    
    ISO -.-> |drives| OPT
    ISO -.-> |drives| EQ
    QUOTE -.-> |drives| ETF
    
    KALSHI --> |affects| SYS
    EQ --> |split into| SYS
    EQ --> |split into| IDIO
    
    style KALSHI fill:#ff6b6b
    style OPT fill:#ffd93d
    style ETF fill:#6bcf7f
    style EQ fill:#4d96ff
    style ISO fill:#ee5a6f
    style QUOTE fill:#95e1d3
```

------------------------------------------------------------------------

## Team Coordination Flow

``` mermaid
graph LR
    subgraph "Greg - CSP Implementation"
        G1[Run Ticker Query<br/>5 min]
        G2[Build Parquet Adapters<br/>15 min]
        G3[Kalshi Kalman Filter<br/>15 min]
        G4[Feature Aggregation<br/>20 min]
        G5[Integration<br/>30 min]
        G6[Dashboard<br/>30 min]
        
        G1 --> G2 --> G3 --> G4 --> G5 --> G6
    end
    
    subgraph "David - Functions & Model"
        D1[Receive Ticker List<br/>from Greg]
        D2[Train ML Model<br/>Oct 5 - Nov 4<br/>25 min]
        D3[Extract Leadership<br/>Functions<br/>20 min]
        D4[Validate on Sample<br/>10 min]
        
        D1 --> D2
        D1 --> D3
        D2 --> D4
        D3 --> D4
    end
    
    subgraph "Anoush - Factor Functions"
        A1[Receive Ticker List<br/>from Greg]
        A2[Factor Calculator<br/>15 min]
        A3[Beta Decomposition<br/>15 min]
        A4[Test Functions<br/>10 min]
        
        A1 --> A2 --> A3 --> A4
    end
    
    G1 -.ticker list.-> D1
    G1 -.ticker list.-> A1
    D4 -.functions.-> G5
    A4 -.functions.-> G5
    
    style G1 fill:#ff6b6b
    style D1 fill:#4d96ff
    style A1 fill:#6bcf7f
```

------------------------------------------------------------------------

## CSP Graph Structure

``` mermaid
graph TB
    subgraph "Main Graph"
        CONFIG[Config Dict<br/>file paths<br/>ticker lists]
        
        CONFIG --> MAIN[csp.graph<br/>main_alphathon_graph]
        
        MAIN --> OUT1[equity_features<br/>basket]
        MAIN --> OUT2[kalshi_filtered<br/>ts]
        MAIN --> OUT3[fama_factors<br/>ts]
        MAIN --> OUT4[leadership_scores<br/>ts]
        MAIN --> OUT5[rnd_surfaces<br/>dict optional]
    end
    
    subgraph "Data Readers - Inside Main"
        TR[Trade Reader]
        QR[Quote Reader]
        OR[Option Reader]
        KR[Kalshi Reader]
    end
    
    subgraph "Feature Nodes - Inside Main"
        FN1[build_equity_features<br/>×40 tickers]
        FN2[kalshi_kalman_filter]
        FN3[compute_fama_factors]
        FN4[David's wrappers]
        FN5[RND extraction optional]
    end
    
    CONFIG --> TR
    CONFIG --> QR
    CONFIG --> OR
    CONFIG --> KR
    
    TR --> FN1
    QR --> FN1
    KR --> FN2
    FN1 --> FN3
    FN1 --> FN4
    OR --> FN5
    
    FN1 --> OUT1
    FN2 --> OUT2
    FN3 --> OUT3
    FN4 --> OUT4
    FN5 --> OUT5
```

------------------------------------------------------------------------

## Feature Engineering Detail

``` mermaid
graph LR
    subgraph "Trade Stream Processing"
        T1[Trade Tick<br/>price, size, conditions]
        T2{Check<br/>condition 53?}
        T3[Tag as ISO]
        T4[Tag as Non-ISO]
        T5{Price vs Mid?}
        T6[dir_ind = +1<br/>Buy]
        T7[dir_ind = -1<br/>Sell]
        T8[dir_ind = 0<br/>Mid]
        
        T1 --> T2
        T2 -->|Yes| T3
        T2 -->|No| T4
        T3 --> T5
        T4 --> T5
        T5 -->|Above| T6
        T5 -->|Below| T7
        T5 -->|Equal| T8
    end
    
    subgraph "1-Second Aggregation"
        BUF[Trade Buffer<br/>accumulate 1s]
        
        T6 --> BUF
        T7 --> BUF
        T8 --> BUF
        
        BUF --> FLOW[Compute Flow<br/>Σ dir_ind × size × price]
        BUF --> SLIP[Compute Slippage<br/>Σ dir_ind × price - mid]
        BUF --> VOL[Compute Volume<br/>Σ size]
        BUF --> CNT[Count Trades<br/>n_trades, n_iso]
        
        FLOW --> ISO_FLOW[iso_flow<br/>non_iso_flow]
        SLIP --> ISO_SLIP[iso_slippage<br/>non_iso_slippage]
        VOL --> ISO_VOL[volume_iso<br/>pct_volume_iso]
        CNT --> ISO_CNT[num_trades_iso<br/>pct_trades_iso]
    end
    
    subgraph "Final Bar"
        BAR[EquityBar1s<br/>15+ features]
        
        ISO_FLOW --> BAR
        ISO_SLIP --> BAR
        ISO_VOL --> BAR
        ISO_CNT --> BAR
    end
```

------------------------------------------------------------------------

## Kalshi Kalman Filter State Machine

``` mermaid
stateDiagram-v2
    [*] --> Initialize: Start
    Initialize --> Predict: New Trade Arrives
    
    state Predict {
        [*] --> TimeUpdate
        TimeUpdate --> p_pred = p_mean
        p_pred --> p_var_pred = p_var + Q
    }
    
    Predict --> Correct: Observation y
    
    state Correct {
        [*] --> ComputeGain
        ComputeGain --> K = p_var_pred / (p_var_pred + R)
        K --> UpdateMean
        UpdateMean --> p_mean = p_pred + K(y - p_pred)
        p_mean --> UpdateVar
        UpdateVar --> p_var = (1 - K) × p_var_pred
    }
    
    Correct --> Innovate: Calculate Innovation
    
    state Innovate {
        [*] --> CalcInnovation
        CalcInnovation --> innovation = y - p_pred
        innovation --> Normalize
        Normalize --> normalized = innovation / sqrt(p_var_pred + R)
    }
    
    Innovate --> Emit: Output KalshiBar1s
    Emit --> Predict: Next Trade
```

------------------------------------------------------------------------

## Leadership Analysis Pipeline

``` mermaid
graph TB
    subgraph "Input: Equity Price Matrix"
        PM[Price Matrix<br/>timestamp × tickers<br/>log-prices or returns]
    end
    
    subgraph "Path A: Raw Returns"
        PM --> VECM1[VECM Estimation<br/>k_ar_diff=1<br/>coint_rank=1]
        VECM1 --> ALPHA1[Extract alpha, Omega]
        ALPHA1 --> APERP1[Compute alpha_perp<br/>SVD]
        APERP1 --> IS1[Information Share<br/>Hasbrouck formula]
        IS1 --> LEAD1[Leadership Scores<br/>RAW]
    end
    
    subgraph "Path B: De-beta'd Returns"
        PM --> DEBETA[Remove Systematic<br/>r_idio = r - β'F]
        DEBETA --> VECM2[VECM on Idio]
        VECM2 --> ALPHA2[Extract alpha, Omega]
        ALPHA2 --> APERP2[Compute alpha_perp]
        APERP2 --> IS2[Information Share]
        IS2 --> LEAD2[Leadership Scores<br/>IDIOSYNCRATIC]
    end
    
    subgraph "Comparison & Interpretation"
        LEAD1 --> COMP{Compare IS Scores}
        LEAD2 --> COMP
        COMP --> |IS_raw > IS_idio| SYS[ISO Flow affects<br/>SYSTEMATIC info]
        COMP --> |IS_idio > IS_raw| IDIO[ISO Flow affects<br/>IDIOSYNCRATIC info]
    end
    
    style LEAD1 fill:#d4edda
    style LEAD2 fill:#cfe2ff
    style SYS fill:#fff3cd
    style IDIO fill:#f8d7da
```

------------------------------------------------------------------------

## RND Extraction & Superimposition

``` mermaid
graph TB
    subgraph "Option Chain Input"
        OPT[VectorizedOptionQuote<br/>strikes, IVs, tte, spot]
    end
    
    subgraph "GPR RND Extraction - Greg's Existing Code"
        GPR1[Gaussian Process<br/>Regression]
        GPR2[Kernel: RBF + Matern]
        GPR3[Optimize Hyperparameters]
        GPR4[Predict on Strike Grid]
        
        OPT --> GPR1
        GPR1 --> GPR2 --> GPR3 --> GPR4
    end
    
    subgraph "RND Output"
        GRID[Strike Grid<br/>1000 points]
        PDF[Probability Density<br/>f S]
        CDF[Cumulative Distribution<br/>F S]
        IV[Smooth IV Curve]
        
        GPR4 --> GRID
        GPR4 --> PDF
        GPR4 --> CDF
        GPR4 --> IV
    end
    
    subgraph "Superimposition Model"
        KALSHI[Kalshi Probability<br/>p = filtered_prob]
        
        PDF --> MODEL["Model: f(S) = p·f1(S) + (1-p)·f0(S)"]
        KALSHI --> MODEL
        
        MODEL --> SOLVE[Regression or<br/>Maximum Entropy]
        SOLVE --> PDF1["f(S when K=1)"]
        SOLVE --> PDF0["f(S when K=0)"]
    end
    
    subgraph "Validation"
        PDF1 --> CHECK["Check: p·PDF1 + (1-p)·PDF0 ≈ PDF"]
        PDF0 --> CHECK
        PDF --> CHECK
        CHECK --> R2[Regression R²]
    end
    
    style PDF fill:#d4edda
    style PDF1 fill:#fff3cd
    style PDF0 fill:#f8d7da
```

------------------------------------------------------------------------

## CBRA Joint Distribution (Stretch Goal)

``` mermaid
graph TB
    subgraph "Inputs: Marginal RNCDs"
        M1[AAPL RND<br/>CDF_1 S]
        M2[MSFT RND<br/>CDF_2 S]
        M3[... D equities]
        M4[SPY RND<br/>CDF_D+1 S]
        M5[QQQ RND<br/>CDF_D+2 S]
        M6[... K ETFs]
    end
    
    subgraph "CBRA Algorithm - Greg's Package"
        DISC[Discretize<br/>n=5000 states]
        INIT[Initialize Matrix<br/>Y: n × D+K]
        RAND[Randomize Columns]
        BLOCKS[Identify Admissible<br/>Blocks]
        OPT[CBRA Optimize<br/>Iterative Swaps]
        
        M1 --> DISC
        M2 --> DISC
        M3 --> DISC
        M4 --> DISC
        M5 --> DISC
        M6 --> DISC
        
        DISC --> INIT --> RAND --> BLOCKS --> OPT
    end
    
    subgraph "Outputs: Joint Distribution"
        JOINT[Joint RND<br/>n states × D+K assets]
        PRICE[Price Any Portfolio<br/>custom weights]
        FAMA_RND[Implied Fama Factor RND<br/>project onto factor space]
        
        OPT --> JOINT
        JOINT --> PRICE
        JOINT --> FAMA_RND
    end
    
    subgraph "Factor Projection"
        JOINT --> PROJ[For each state i:<br/>F_i = Σ w_j × Asset_j_i]
        WEIGHTS[Fama Weights] --> PROJ
        PROJ --> F_DIST[Factor Distribution<br/>f MKT-RF, f SMB, ...]
    end
    
    style JOINT fill:#d1ecf1
    style FAMA_RND fill:#fff3cd
```

------------------------------------------------------------------------

## Dashboard Update Mechanism

``` mermaid
sequenceDiagram
    participant CSP as CSP Graph
    participant COLLECT as Output Collector
    participant PANEL as Panel Dashboard
    participant USER as User Browser
    
    CSP->>COLLECT: Stream tick (EquityBar1s)
    CSP->>COLLECT: Stream tick (KalshiBar1s)
    CSP->>COLLECT: Stream tick (FamaFactors)
    
    COLLECT->>COLLECT: Buffer for 1 second
    
    COLLECT->>PANEL: Update data structures
    
    PANEL->>PANEL: Refresh widgets
    
    PANEL->>USER: Push updates (WebSocket)
    
    Note over CSP,USER: Real-time updates at 1Hz
    
    USER->>PANEL: Select ticker/timeframe
    PANEL->>PANEL: Filter & re-render
    PANEL->>USER: Updated visualization
```

------------------------------------------------------------------------

## Execution Timeline Gantt

``` mermaid
gantt
    title Alphathon Q1 Execution Timeline (2.5 Hours)
    dateFormat HH:mm
    axisFormat %H:%M
    
    section Greg
    Ticker Query           :crit, g1, 00:00, 5m
    Team Coordination      :crit, g2, 00:05, 10m
    Parquet Adapters       :crit, g3, 00:15, 15m
    Kalshi Filter          :crit, g4, 00:30, 15m
    Feature Aggregation    :crit, g5, 00:45, 15m
    RND Wire-up            :g6, 01:00, 15m
    Integration            :crit, g7, 01:15, 30m
    Dashboard              :crit, g8, 01:45, 15m
    CSV Generation         :g9, 02:00, 15m
    
    section David
    Receive Spec           :milestone, d1, 00:05, 0m
    Model Training         :d2, 00:15, 25m
    Extract Functions      :d3, 00:15, 20m
    Validate               :d4, 00:40, 20m
    Deliver Functions      :milestone, d5, 01:00, 0m
    
    section Anoush
    Receive Spec           :milestone, a1, 00:05, 0m
    Factor Calculator      :a2, 00:15, 15m
    Beta Decomposition     :a3, 00:30, 15m
    Deliver Functions      :milestone, a4, 00:45, 0m
    
    section All
    Paper Draft            :p1, 02:15, 30m
    Final Packaging        :crit, p2, 02:45, 15m
```

------------------------------------------------------------------------

## Decision Tree: Scope Management

``` mermaid
graph TD
    START[Start: 00:00] --> Q1{Ticker Query<br/>Complete?}
    Q1 -->|Yes| Q2[00:05: Message Team]
    Q1 -->|No - stuck| FALLBACK1[Use hardcoded<br/>top 20 tickers]
    
    Q2 --> M1{00:30:<br/>Adapters Working?}
    M1 -->|Yes| M2[Continue: Features]
    M1 -->|No| FALLBACK2[Simplify:<br/>1 ticker test case]
    
    M2 --> M3{01:00:<br/>David/Anoush<br/>Delivered?}
    M3 -->|Yes| M4[Continue: Integration]
    M3 -->|No| FALLBACK3[Use simple<br/>correlation fallbacks]
    
    M4 --> M5{01:30:<br/>Core Graph<br/>Working?}
    M5 -->|Yes| M6{Time<br/>Remaining?}
    M5 -->|No| FALLBACK4[Debug core<br/>Cut RND/CBRA]
    
    M6 -->|>45 min| M7[Add: RND + Dashboard]
    M6 -->|30-45 min| M8[Add: Dashboard only]
    M6 -->|<30 min| M9[Static plots only]
    
    M7 --> END[02:30: Generate Results]
    M8 --> END
    M9 --> END
    
    END --> FINAL{02:45:<br/>Paper Done?}
    FINAL -->|Yes| SUBMIT[03:00: Submit]
    FINAL -->|No| RUSH[Rush minimal paper]
    RUSH --> SUBMIT
    
    style START fill:#d4edda
    style SUBMIT fill:#d4edda
    style FALLBACK1 fill:#f8d7da
    style FALLBACK2 fill:#f8d7da
    style FALLBACK3 fill:#f8d7da
    style FALLBACK4 fill:#f8d7da
```

------------------------------------------------------------------------

**USAGE**: Reference these diagrams for: - System understanding - Team coordination - Paper figures (include Architecture, Information Flow, Team Coordination) - Debugging (follow data flow paths)

**NEXT**: Start executing! Run ticker query and begin CSP implementation! 🚀