# DIP_Affine_Transform
2019 MMI Lab. DIP 세미나 
Image Affine Transform, Color Transform – RGB to HIS, and Spatial Domain Filtering

1.	발표 내용 - 교재(Rafael C. Gonzalez, Richard E. Woods, “Digital Image Processing”, 4th global edition)의 Ch.3참고
각 구현 내용을 함수나 class 별로 나눠서 함수의 재사용이 가능하도록 구현합니다. 이번 주차의 구현 내용은 다음과 같습니다.

1.	Intensity Transform
Histogram equalization을 제외한 모든 intensity transform(negative와 gamma, power 등)을 구현합니다.

2.	Affine Transformation
Image zooming과 shrinking, rotation 등 다양한 affine transform을 직접 코드로 구현합니다. 읽어온 image가 discrete signal인 점을 고려하여 어떤 방식으로 구현하는 것이 정확할 지 스스로 생각하는 것이 중요합니다. 예를 들어 scaling할 때 생기는 빈 공간을 어떤 interpolation으로 채우는 지에 따라 결과가 달라집니다.

3.	Color Transform
기본적으로 RGB to HIS transform과 HIS to RGB transform을 구현합니다. 여유가 된다면 다른 color transform을 찾고 구현해봅니다.
