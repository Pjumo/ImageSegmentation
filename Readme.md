# 1. Backbone 모델 선정

- BackBone : `deeplabv3plus_resnet101(DeepLabV3_ResNet101_Weights.COCO_WITH_VOC_LABELS_V1)`

![Untitled](1%20Backbone%20%E1%84%86%E1%85%A9%E1%84%83%E1%85%A6%E1%86%AF%20%E1%84%89%E1%85%A5%E1%86%AB%E1%84%8C%E1%85%A5%E1%86%BC%204f180a9cc5cf4768a25b4edb46e46a75/Untitled.png)

- deeplabv3plus_resnet50

![Untitled](1%20Backbone%20%E1%84%86%E1%85%A9%E1%84%83%E1%85%A6%E1%86%AF%20%E1%84%89%E1%85%A5%E1%86%AB%E1%84%8C%E1%85%A5%E1%86%BC%204f180a9cc5cf4768a25b4edb46e46a75/Untitled%201.png)

- deeplabv3plus_resnet101

![Untitled](1%20Backbone%20%E1%84%86%E1%85%A9%E1%84%83%E1%85%A6%E1%86%AF%20%E1%84%89%E1%85%A5%E1%86%AB%E1%84%8C%E1%85%A5%E1%86%BC%204f180a9cc5cf4768a25b4edb46e46a75/Untitled%202.png)

|  | deeplabv3plus_resnet50  | deeplabv3plus_resnet101 | deeplabv3plus_resnet101_COCO |
| --- | --- | --- | --- |
| PA | 86.35% | 89.88% | 94.31% |
| miou | 43.56% | 53.42% | 72.93% |

# 2. Parameter 설정

- Loss function
    - **Focal loss** : 주로 클래스 불균형 문제를 해결하기 위해 설계된 손실 함수,  어려운 예제(즉, 잘못 분류된 예제)에 더 많은 가중치를 부여하고, 쉬운 예제(즉, 올바르게 분류된 예제)에 덜 가중치를 부여함으로써 모델이 어려운 문제를 더 집중적으로 학습하도록 한다.
        
        ```python
        class FocalLoss(nn.Module):
            def __init__(self, alpha=1, gamma=0, size_average=True, ignore_index=255):
                super(FocalLoss, self).__init__()
                self.alpha = alpha
                self.gamma = gamma
                self.ignore_index = ignore_index
                self.size_average = size_average
        
            def forward(self, inputs, targets):
                ce_loss = F.cross_entropy(
                    inputs, targets, reduction='none')
                pt = torch.exp(-ce_loss)
                focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
                if self.size_average:
                    return focal_loss.mean()
                else:
                    return focal_loss.sum()
        
        ```
        
    - **Combined loss** : CrossEntrophy loss와 Dice loss를 함께 사용하여 분할 경계의 정밀도 및 픽셀 단위의 정확도를 향상시킨다.
        
        ```python
        def cross_entropy_loss(logits, targets):
            loss = F.cross_entropy(logits, targets)
            return loss
        
        def dice_coefficient(logits, targets, smooth=1):
            probs = torch.sigmoid(logits)
            num = (probs * targets).sum()
            denom = probs.sum() + targets.sum() + smooth
            dice = (2 * num + smooth) / denom
            return dice
        
        def dice_loss(logits, targets, smooth=1):
            loss = 1 - dice_coefficient(logits, targets, smooth)
            return loss
        
        def combined_loss(logits, targets, alpha=0.75):
            ce_loss = cross_entropy_loss(logits, targets)
            d_loss = dice_loss(logits, targets)
            loss = alpha * ce_loss + (1 - alpha) * d_loss
            return loss
        ```
        
    
- Scheduler
    - **ReduceLROnPlateau :** 학습률 감소 기법 중 하나로, 검증 손실(validation loss)이 더 이상 개선되지 않을 때 학습률을 동적으로 감소시켜 모델의 학습을 돕는 기법입니다.
        
        ```
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, 
        						mode='min', factor=0.1, patience=5, verbose=True)
        ```
        
- optimizer
    - ADAM : 학습률을 자동으로 조정하면서 파라미터를 최적화하는 기법이다.
        
        ```jsx
        torch.optim.Adam(self.model.parameters(), lr=0.00001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)
        ```
        

# 3. CRF 적용

- CRF(Conditional Random Field) : 구조 예측문제에 자주 사용되는 통계적 모델, 후처리 과정에서 사용하면 연속된 데이터의 레이블을 좀 더 정확하게 예측할 수 있다.
    
    ![스크린샷 2024-05-29 153129.png](1%20Backbone%20%E1%84%86%E1%85%A9%E1%84%83%E1%85%A6%E1%86%AF%20%E1%84%89%E1%85%A5%E1%86%AB%E1%84%8C%E1%85%A5%E1%86%BC%204f180a9cc5cf4768a25b4edb46e46a75/%25EC%258A%25A4%25ED%2581%25AC%25EB%25A6%25B0%25EC%2583%25B7_2024-05-29_153129.png)
    
    ![스크린샷 2024-05-29 153151.png](1%20Backbone%20%E1%84%86%E1%85%A9%E1%84%83%E1%85%A6%E1%86%AF%20%E1%84%89%E1%85%A5%E1%86%AB%E1%84%8C%E1%85%A5%E1%86%BC%204f180a9cc5cf4768a25b4edb46e46a75/%25EC%258A%25A4%25ED%2581%25AC%25EB%25A6%25B0%25EC%2583%25B7_2024-05-29_153151.png)
    

### Dense_CRF 적용 모습(normalization)

![스크린샷 2024-06-08 233123.png](1%20Backbone%20%E1%84%86%E1%85%A9%E1%84%83%E1%85%A6%E1%86%AF%20%E1%84%89%E1%85%A5%E1%86%AB%E1%84%8C%E1%85%A5%E1%86%BC%204f180a9cc5cf4768a25b4edb46e46a75/%25EC%258A%25A4%25ED%2581%25AC%25EB%25A6%25B0%25EC%2583%25B7_2024-06-08_233123.png)

![스크린샷 2024-06-09 210014.png](1%20Backbone%20%E1%84%86%E1%85%A9%E1%84%83%E1%85%A6%E1%86%AF%20%E1%84%89%E1%85%A5%E1%86%AB%E1%84%8C%E1%85%A5%E1%86%BC%204f180a9cc5cf4768a25b4edb46e46a75/%25EC%258A%25A4%25ED%2581%25AC%25EB%25A6%25B0%25EC%2583%25B7_2024-06-09_210014.png)

![스크린샷 2024-06-09 210030.png](1%20Backbone%20%E1%84%86%E1%85%A9%E1%84%83%E1%85%A6%E1%86%AF%20%E1%84%89%E1%85%A5%E1%86%AB%E1%84%8C%E1%85%A5%E1%86%BC%204f180a9cc5cf4768a25b4edb46e46a75/%25EC%258A%25A4%25ED%2581%25AC%25EB%25A6%25B0%25EC%2583%25B7_2024-06-09_210030.png)

→ 학습이 진행될 수록 이미지가 망가지는 모습을 볼 수 있다.

### CRF적용(none-normalization)

![스크린샷 2024-06-10 135108.png](1%20Backbone%20%E1%84%86%E1%85%A9%E1%84%83%E1%85%A6%E1%86%AF%20%E1%84%89%E1%85%A5%E1%86%AB%E1%84%8C%E1%85%A5%E1%86%BC%204f180a9cc5cf4768a25b4edb46e46a75/%25EC%258A%25A4%25ED%2581%25AC%25EB%25A6%25B0%25EC%2583%25B7_2024-06-10_135108.png)

![스크린샷 2024-06-10 135223.png](1%20Backbone%20%E1%84%86%E1%85%A9%E1%84%83%E1%85%A6%E1%86%AF%20%E1%84%89%E1%85%A5%E1%86%AB%E1%84%8C%E1%85%A5%E1%86%BC%204f180a9cc5cf4768a25b4edb46e46a75/%25EC%258A%25A4%25ED%2581%25AC%25EB%25A6%25B0%25EC%2583%25B7_2024-06-10_135223.png)

![스크린샷 2024-06-10 135307.png](1%20Backbone%20%E1%84%86%E1%85%A9%E1%84%83%E1%85%A6%E1%86%AF%20%E1%84%89%E1%85%A5%E1%86%AB%E1%84%8C%E1%85%A5%E1%86%BC%204f180a9cc5cf4768a25b4edb46e46a75/%25EC%258A%25A4%25ED%2581%25AC%25EB%25A6%25B0%25EC%2583%25B7_2024-06-10_135307.png)

![Untitled](1%20Backbone%20%E1%84%86%E1%85%A9%E1%84%83%E1%85%A6%E1%86%AF%20%E1%84%89%E1%85%A5%E1%86%AB%E1%84%8C%E1%85%A5%E1%86%BC%204f180a9cc5cf4768a25b4edb46e46a75/Untitled%203.png)

|  | normalized | none-normalized |
| --- | --- | --- |
| PA | 94.63% | 94.39% |
| miou | 76.17% | 76.32% |

# 4. New Idea

- 너무 정확한 경계 표시로 인해 내부의 빈 공간 때문에 miou 성능이 떨어지는 거 아닌가??
    
    → CRF의 적용 전 이미지와 후처리된 CRF의 이미지를 합쳤을 때 성능이 증가하지 않을까??
    
    - normalization
        
        ![Untitled](1%20Backbone%20%E1%84%86%E1%85%A9%E1%84%83%E1%85%A6%E1%86%AF%20%E1%84%89%E1%85%A5%E1%86%AB%E1%84%8C%E1%85%A5%E1%86%BC%204f180a9cc5cf4768a25b4edb46e46a75/Untitled%204.png)
        
        ![Untitled](1%20Backbone%20%E1%84%86%E1%85%A9%E1%84%83%E1%85%A6%E1%86%AF%20%E1%84%89%E1%85%A5%E1%86%AB%E1%84%8C%E1%85%A5%E1%86%BC%204f180a9cc5cf4768a25b4edb46e46a75/Untitled%205.png)
        
        ![Untitled](1%20Backbone%20%E1%84%86%E1%85%A9%E1%84%83%E1%85%A6%E1%86%AF%20%E1%84%89%E1%85%A5%E1%86%AB%E1%84%8C%E1%85%A5%E1%86%BC%204f180a9cc5cf4768a25b4edb46e46a75/Untitled%206.png)
        
    - none-normalization
        
        ![스크린샷 2024-06-12 025059.png](1%20Backbone%20%E1%84%86%E1%85%A9%E1%84%83%E1%85%A6%E1%86%AF%20%E1%84%89%E1%85%A5%E1%86%AB%E1%84%8C%E1%85%A5%E1%86%BC%204f180a9cc5cf4768a25b4edb46e46a75/%25EC%258A%25A4%25ED%2581%25AC%25EB%25A6%25B0%25EC%2583%25B7_2024-06-12_025059.png)
        
        ![스크린샷 2024-06-12 025338.png](1%20Backbone%20%E1%84%86%E1%85%A9%E1%84%83%E1%85%A6%E1%86%AF%20%E1%84%89%E1%85%A5%E1%86%AB%E1%84%8C%E1%85%A5%E1%86%BC%204f180a9cc5cf4768a25b4edb46e46a75/%25EC%258A%25A4%25ED%2581%25AC%25EB%25A6%25B0%25EC%2583%25B7_2024-06-12_025338.png)
        
        ![Untitled](1%20Backbone%20%E1%84%86%E1%85%A9%E1%84%83%E1%85%A6%E1%86%AF%20%E1%84%89%E1%85%A5%E1%86%AB%E1%84%8C%E1%85%A5%E1%86%BC%204f180a9cc5cf4768a25b4edb46e46a75/Untitled%207.png)
        
        ![Untitled](1%20Backbone%20%E1%84%86%E1%85%A9%E1%84%83%E1%85%A6%E1%86%AF%20%E1%84%89%E1%85%A5%E1%86%AB%E1%84%8C%E1%85%A5%E1%86%BC%204f180a9cc5cf4768a25b4edb46e46a75/Untitled%208.png)
        

- 하루만에 결과를 뽑아야되지만 학습시간이 길어서 원하는 결과를 얻지 못해 아쉬웠다.

### 최종 결과

 `Deeplabv3plus_resnet101(DeepLabV3_ResNet101_Weights.COCO_WITH_VOC_LABELS_V1)+Dense CRF`

| PA | 94.39% |
| --- | --- |
| miou | 76.32% |
- 코드
    
    

[GitHub - Pjumo/ImageSegmentation: ImageSegmentation for VOC2012 datasets](https://github.com/Pjumo/ImageSegmentation)