
python inference_seg_unetformer.py \
-i ../data/segmentation/Turkey/Islahiye/pre/test/images \
-c config/teq/seg_unetformer.py \
-o fig_results/teq/unetformer-w-pred_offsets \
 -ph 512 -pw 512 -b 4 -d "building"
