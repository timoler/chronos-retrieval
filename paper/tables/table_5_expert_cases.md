# Table 5: 专家评价与修订案例（精选）

| Dataset | Expert | Round | Pool size | Decision | Reward | Coverage | Mean pool reward | Positive-fold ratio | Replaced | Mechanism (abridged) |
|---|---|---|---|---|---|---|---|---|---|---|
| etth1 | E1 | 2 | 5 | KEEP | 2 | 0.449 | +0.1757 | 0.75 | True | High oil temperature usually indicates a heavy-load or high-ambient thermal state whose... |
| etth1 | E4 | 0 | 8 | KEEP | 2 | 0.454 | +0.0029 | 0.75 | False | High recent volatility implies rapidly varying load and intermittent heat input, so onl... |
| etth1 | E7 | 2 | 5 | KEEP | 1 | 0.454 | +0.2272 | 0.50 | True | High recent volatility indicates an unstable thermal disturbance regime, where only sho... |
| etth1 | E8 | 2 | 5 | UNSUPPORTED | 0 | 0.000 | +0.1657 | 0.50 | True | Low recent autocorrelation means the recent sequence is nearly unpredictable, so long h... |
| electricity | E2 | 2 | 3 | REVISE | -1 | 0.458 | -0.0877 | 0.50 | True | When recent volatility is high, volatility regimes often have season-specific drivers, ... |
| electricity | E3 | 1 | 5 | KEEP | 2 | 0.061 | +0.0271 | 1.00 | True | Morning ramps are dominated by a consistent start-of-day load pickup, so high recent sl... |
| electricity | E8 | 2 | 3 | REJECT | -2 | 0.283 | -0.2185 | 0.00 | True | Persistent high-load plateaus are primarily shaped by the workday/non-working-day deman... |
| traffic | E1 | 0 | 8 | KEEP | 2 | 0.178 | +0.0062 | 0.75 | False | On workday mornings, occupancy is dominated by the regular commute, so historical workd... |
| traffic | E5 | 2 | 3 | REJECT | -2 | 0.440 | -0.0197 | 0.00 | True | High recent slope magnitude indicates a strong directional regime; matching that slope ... |

精选保留/修订/剔除案例（每条取该 lineage 最后一轮评价记录）。Reward∈{2,1,0,−2,−1} 对应 KEEP/KEEP/UNSUPPORTED/REJECT/REVISE。Coverage=TRAIN 激活覆盖率（阈值 0.5）。留一贡献对应当轮池组成（屏蔽该专家嵌入+激活后重训，不删除槽位）。完整 per-expert/per-round 记录见 results/reward_feedback_critic_v2_full_cross_dataset/<ds>/pool_reward_history.csv。
