# python inference.py \
# -i ../data/xBD/test/images \
# -c config/xBD/unetformer.py \
# -o fig_results/xbd/unetformer \
# -t 'lr' -ph 512 -pw 512 -b 4 -d "building"

# python inference_emd.py \
# -i ../data/segmentation/Turkey/Islahiye/pre/test/images \
# -c config/xBD_Teq/unetformer.py \
# -o fig_results/xbd_teq/emd \
# -t 'lr' -ph 512 -pw 512 -b 4 -d "building"

# python inference_advent.py \
# -i ../data/segmentation/Turkey/Islahiye/pre/test/images \
# -c config/xBD_Teq/unetformer.py \
# -o fig_results/xbd_teq/unetformer \
# -t 'lr' -ph 512 -pw 512 -b 4 -d "building"

# python inference_advent_emd.py \
# -i ../data/segmentation/Turkey/Islahiye/pre/test/images \
# -c config/xBD_Teq/unetformer.py \
# -o fig_results/xbd_teq/advent_emd \
# -t 'lr' -ph 512 -pw 512 -b 4 -d "building"

python inference_seg_deeplab.py \
-i ../data/segmentation/Turkey/Islahiye/pre/test/images \
-c config/teq/seg_deeplab.py \
-o fig_results/teq/deeplab-w-pred_offsets \
 -ph 512 -pw 512 -b 4 -d "building"

# python inference_offset.py \
# -i ../data/segmentation/Turkey/Islahiye/pre/test/images \
# -c config/teq/v4_offset.py \
# -o fig_results/teq/offset-v4-pretrain-xbd-RSB-predict-object \
#  -ph 512 -pw 512 -b 4 -d "building"