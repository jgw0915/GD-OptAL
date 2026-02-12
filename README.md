# Empirical Study of Gradient-Based Optimizers for OCT Binary Classification  
### From Theoretical Foundations to Optimizer Enhancement (Adam Variants + AutoML)

**Project Type**: 
- Machine Learning/Deep Learning Optimization Research  

**Related Materials**:  
- [機器/深度學習-基礎數學(三):梯度最佳解相關算法](https://chih-sheng-huang821.medium.com/%E6%A9%9F%E5%99%A8%E5%AD%B8%E7%BF%92-%E5%9F%BA%E7%A4%8E%E6%95%B8%E5%AD%B8-%E4%B8%89-%E6%A2%AF%E5%BA%A6%E6%9C%80%E4%BD%B3%E8%A7%A3%E7%9B%B8%E9%97%9C%E7%AE%97%E6%B3%95-gradient-descent-optimization-algorithms-b61ed1478bd7)
- "The Empirical Comparison of Deep Neural Network Optimizers for Binary Classification of OCT Images"

---

## Project Motivation

在深度學習模型中，**Optimizer 的選擇與超參數設定**往往直接影響模型的收斂速度、穩定性與泛化能力。

本專案從兩個面向出發：

1. **理論基礎建構**  
   系統化整理 Gradient Descent、SGD、Momentum、Adagrad、RMSProp、Adam 等演算法的數學原理與更新機制（期末報告內容）。

2. **實證比較分析**  
   根據論文 *The Empirical Comparison of Deep Neural Network Optimizers for Binary Classification of OCT Images*，分析不同優化器於醫療影像（二元 OCT 分類）任務中的實際表現。

在完成理論與實證研究後，我進一步提出：

> 使用 **Adam 變體（AdamW、AMSGrad）與自動化超參數調優（AutoML）策略** 進行再優化，以提升泛化能力與穩定性。

---

## 📖 Part I — Theoretical Foundation: Gradient-Based Optimization

（內容整理自期末報告 :contentReference[oaicite:2]{index=2}）

### Gradient Descent (GD)

目標：
$\theta_{t+1} = \theta_t - \gamma \nabla J(\theta_t)]$

- 需使用完整資料集
- 計算成本高
- 易陷入 Local Minimum

---

### SGD (Stochastic Gradient Descent)

- Mini-batch 更新
- 收斂速度提升
- 對 learning rate 極度敏感

---

### Momentum

$v_t = \beta v_{t-1} + (1-\beta) g_t$

- 減少震盪
- 改善 saddle point 問題
- 仍需手動設定超參數

---

### Adagrad

- 累積歷史梯度平方
- 對稀疏特徵表現佳
- 學習率單調下降（可能過早停滯）

---

### RMSProp

- 使用指數移動平均
- 解決 Adagrad 學習率過快衰減問題
- 對 decay rate 敏感

---

### Adam (Adaptive Moment Estimation)

結合：
- Momentum（一階動量）
- RMSProp（二階動量）

特性：
- 收斂快速
- 訓練穩定
- 在多數深度模型中為預設選擇

但問題：
- 理論收斂證明較弱
- 在某些情況下可能震盪
- 泛化能力不一定最佳

---

## Part II — Empirical Comparison on OCT Binary Classification

（分析自論文 : "THE EMPIRICAL COMPARISON OF DEEP NEURAL NETWORK OPIMIZERS FOR BINARY CLASSIFICATION OF OCT IMAGES" ）

### Dataset

- OCTID dataset
- Normal vs AMD
- Image size: 224 × 224 × 3
- 10 epochs training
- Loss: Binary Cross Entropy

---

### Models Compared

- CNN
- DNN
- VGG16

---

### Optimizers Compared

- SGD
- SGD + Momentum
- Adagrad
- RMSProp
- Adam

Learning rates:
- 0.001
- 0.0001
- 0.00001

---

## Key Findings

### Adam 表現最佳

- 在多數 learning rate 下達到最低 train/test loss
- 多數模型 training accuracy = 100%
- 在 VGG16 中測試準確率最高可達 100%

### RMSProp 次佳

- 穩定
- 部分情況測試準確率接近 Adam
- 但初期 loss 較高

### SGD 系列表現最差

- 收斂慢
- 對 learning rate 高度敏感
- 易震盪

---

## My Research Extension

在完成理論推導與實證比較後，我提出以下進一步優化方向：

---

# Adam 變體優化

## AdamW（Decoupled Weight Decay）

**問題**：
Adam 原始實作中，L2 regularization 與 adaptive learning rate 耦合，可能影響泛化。

**解法**：
AdamW 將 weight decay 與梯度更新分離。

**優勢**：
- 提升泛化能力
- 在 vision tasks 中效果顯著優於 Adam

---

## AMSGrad

**問題**：
Adam 的二階動量估計可能導致不收斂。

**AMSGrad 修正**：
$\hat{v}_t = \max(\hat{v}_{t-1}, v_t)$

優勢：
- 理論上保證收斂
- 避免震盪

---

# AutoML for Hyperparameter Optimization

論文中學習率為固定設定（0.001 / 0.0001 / 0.00001）。

改進方向：

- Bayesian Optimization
- Optuna / Ray Tune
- Learning rate scheduling
- β₁, β₂ 自動搜尋
- Weight decay tuning

優點：

- 避免人工 trial-and-error
- 提升泛化能力
- 降低 overfitting 風險

---

# Proposed Experimental Pipeline

1. 使用 AdamW / AMSGrad 替代 Adam
2. 導入 cosine annealing 或 warmup schedule
3. 使用 Bayesian Optimization 搜尋：
   - learning rate
   - β₁, β₂
   - weight decay
4. Cross-validation 驗證泛化能力

---

# Research Insight

本專案不僅比較 optimizer。

而是：

> 從數學原理 → 到醫療影像實證 → 再到優化演算法改進策略

這種完整研究流程，體現我在：

- Optimization 理論理解
- 深度學習實驗設計
- 模型泛化能力思考
- AutoML 流程設計能力

上的整合能力。

---

# Future Research Direction

- Optimizer + Hardware-aware training
- Mixed precision + optimizer interaction
- Adaptive optimizer behavior under low-data regime
- Post-quantum ML model optimization

---

# Conclusion

本研究顯示：

- Adam 在 OCT 二元分類中表現最佳
- 但仍存在泛化與理論收斂問題
- 透過 AdamW / AMSGrad + AutoML，可進一步提升模型穩定性與泛化能力

完整研究流程：

理論 → 實證 → 改進 → 再優化

---

