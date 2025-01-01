# card_segmentation

## Yêu cầu

Anaconda Navigator (Không cần thiết nếu không muốn tạo môi trường ảo)

> Nếu dùng GPU:

- Cuda Toolkit 
- Pytorch. Chọn compute platform tương ứng với phiên bản Cuda Toolkit

> Nếu không dùng GPU:

- Pytorch. Chọn compute platform là CPU

## Clone github
```
git clone https://github.com/hoanglqd-a1/card_classification.git
```

Tạo môi trường ảo (Có thể bỏ qua)
```
conda create -n card_classification python=3.10.15
conda activate card_classification
```

Tải các thư viện liên quan
```
cd card_segmentation
pip install -r requirements.txt
```
## Tải dataset
Có thể bỏ qua bước này nếu không muốn train mô hình

- Tạo folder card_segmentation/dataset

```
cd card_segmentation
python prepare_dataset.py
```

## Train mô hình

```
python train.py
```

Nếu muốn train tiếp từ mô hình đã lưu (pretrained/model_checkpoint.pt)
```
python train.py --resumeTraining=True
```

## Thử nghiệm
```
python test.py test/sample1.png --output_mask=test/output_mask.png --output_prediction=test/output_pred.png --model=./pretrained/model_final.pt
```
