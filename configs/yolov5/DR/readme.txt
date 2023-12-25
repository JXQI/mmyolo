# 划分数据集
python tools/misc/coco_split.py --json data/arch/result.json --out-dir data/arch/ --ratios 0.8 0.2

#　训练
python tools/train.py configs/yolov5/DR/yolov5_s-v61_syncbn_fast_8xb16-300e_coco.py

# 测试(显示的左图是ground truth, 右图是预测结果)
python tools/test.py ./configs/yolov5/DR/yolov5_s-v61_syncbn_fast_8xb16-300e_coco.py 
    ./work_dirs/yolov5_s-v61_syncbn_fast_8xb16-300e_coco/best_coco_bbox_mAP_epoch_300.pth --show

# 简单对比了一下AP很差，符合预期，因为MA可能太小了，导致计算的结果差，后续更换评价指标，目前先主观判断一下


# 拱环
# 训练
python tools/train.py configs/yolov5/DR/yolov5_ins_s-v61_syncbn_fast_8xb16-300e_coco_instance.py
# 测试
ython tools/test.py configs/yolov5/DR/yolov5_ins_s-v61_syncbn_fast_8xb16-300e_coco_instance.py 
    ./work_dirs/yolov5_ins_s-v61_syncbn_fast_8xb16-300e_coco_instance/best_coco_bbox_mAP_epoch_300.pth