# Chinese and Japanese handwriting generator using CycleGAN

## Prepare data

```bash
python data/prepare_data/prepare_casia.py --source data/sources/1252-c.gnt --font data/fonts/simhei.ttf --fontSize 116 --offset 6 --percent 100 --ratioA 0.9 --ratioB 0.9
```

## Training

```bash
python train.py --dataroot data/datasets/1252-c_116_6_0.9/ --name 1252_dense5 --gpu_ids 2 --which_direction BtoA --loadSize 128 --fineSize 128 --which_model_netG densenet_5blocks --no_dropout --input_nc 1 --output_nc 1
```

## Testing

```bash
python test.py --dataroot data/datasets/1252-c_116_6_0.9/ --name  1252_dense5 --gpu_ids 2  --which_direction BtoA --loadSize 128 --fineSize 128 --which_model_netG densenet_5blocks --no_dropout --which_epoch 185,190,195,200 --how_many 100
```

## Generate a text line

```bash
python test_line.py --name 1244_dense5 --font data/fonts/simhei.ttf --font_size 116 --offset 6 --which_model_netG densenet_5blocks --no_dropout --gpu_ids -2 --input_nc 1 --output_nc 1 --which_direction BtoA --loadSize 128 --fineSize 128 --which_epoch 200
```
