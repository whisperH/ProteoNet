# for ProteoNet training of five fold cross-validation
arr1=("MS_ResWeightedPartNet50_FL")
for subtest in {4..4}
do
  for method in "${arr1[@]}"
  do
    CUDA_VISIBLE_DEVICES=1 python tools/train.py models/resnet/${method}.py --kflod-validation ${subtest}
  done
done

# for mobilenet training of five fold cross-validation
arr1=("MS_WeightedPart_mobilenet_v2_")
for subtest in {0..4}
do
  for method in "${arr1[@]}"
  do
    CUDA_VISIBLE_DEVICES=1 python tools/train.py models/mobilenet/${method}.py --kflod-validation ${subtest}
  done
done
