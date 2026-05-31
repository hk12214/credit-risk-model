# Bati Bank - Alternative Data Credit Risk Probability Model

## Credit Scoring Business Understanding

### 1. Basel II Compliance & Model Interpretability
The Basel II Accord regulates international banking infrastructure to ensure financial institutions hold capital reserves proportional to their operational and credit risks. Under the **Internal Ratings-Based (IRB)** approach of Basel II, banks are permitted to use internal statistical models to estimate credit risk parameters: Probability of Default ($PD$), Loss Given Default ($LGD$), and Exposure at Default ($EAD$).

This framework creates a strict requirement for high model interpretability and exhaustive documentation:
* **Regulatory Auditability:** Because $PD$ models directly impact capital reserve requirements, regulators (such as the National Bank of Ethiopia) must be able to audit every parameter. "Black-box" models that cannot explain *why* a specific score was assigned fail compliance audits.
* **Capital Adequacy Justification:** Documentation must mathematically prove that the model accounts for long-term economic downturns, ensuring the bank isn't underestimating risk and exposing itself to insolvency.
* **Discrimination and Transparency:** Bati Bank must prove that its model does not use variables that violate fair lending laws. Fully explainable features prevent hidden proxy biases against protected socioeconomic indicators.

### 2. The Necessity and Risks of a Proxy Target Variable
The raw transaction data from the Xente eCommerce platform contains consumer behavioral metrics and transaction fraud flags, but it **lacks a historical record of credit performance or actual default events** (e.g., missed 90-day loan payments). Because a true "default" label does not exist, we must construct a **Proxy Target Variable** (using Recency, Frequency, and Monetary metrics via customer segmentation) to simulate credit risk.

While necessary to establish this new Buy-Now-Pay-Later (BNPL) line of business, proxy-based prediction introduces distinct business risks:
* **Target Misalignment (Basis Risk):** A consumer who is classified as "high-risk" due to low platform engagement (low frequency or spending) might actually be a highly creditworthy individual who simply prefers using cash or competing platforms. Conversely, an active shopper could be over-leveraged and prone to default, leading to **False Negatives**.
* **Label Noise:** Transitioning behavioral activity into an indicator of financial distress introduces systematic noise. Training a machine learning model on an unverified proxy label can cause high **False Positive** rates, causing Bati Bank to turn away profitable, safe customers.
* **Concept Drift Amplification:** Consumer habits on an eCommerce app can change quickly due to seasonal marketing campaigns, app design changes, or inventory issues. This shift can cause our proxy label to drift rapidly, even if there has been no actual change in the underlying macroeconomic credit risk environment.

### 3. Model Architecture Trade-Offs in Regulated Finance

Choosing a model for production requires balancing mathematical performance against regulatory acceptance. The table below outlines these core trade-offs:

| Evaluation Dimension | Simple & Interpretable (e.g., Logistic Regression + WoE) | High-Performance (e.g., Gradient Boosting / XGBoost) |
| :--- | :--- | :--- |
| **Explainability** | **Extremely High.** Using Weight of Evidence (WoE) conversion allows each feature coefficient to map directly to a clear credit scorecard point distribution that risk teams and regulators can easily review. | **Low.** Relies on complex, non-linear ensembles of decision trees. Explaining decisions requires secondary frameworks like SHAP or LIME, which provide approximations rather than direct lookups. |
| **Predictive Power** | **Moderate.** Assumes linear relationships between log-odds of risk and features. It may struggle to capture complex, multi-variable interactions present in alternative digital data. | **High.** Strong at discovering non-linear correlations, handling missing data naturally, and identifying complex interactions across behavioral features. |
| **Regulatory Approval** | **Streamlined.** Smooth paths to compliance due to decades of statistical precedent and complete mathematical transparency. | **Challenging.** Requires extensive documentation, extra validation pipelines, and rigorous proof of safety and fairness to pass stringent internal risk committees. |
| **Operational Maintenance** | **Low.** Lightweight, highly stable over time, and computationally inexpensive to run in real-time inference environments. | **High.** Susceptible to overfitting on minor behavioral changes, requiring structured MLOps pipelines (like MLflow) to handle retraining and drift monitoring. |