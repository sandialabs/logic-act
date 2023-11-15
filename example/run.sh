PAR="--num_head 8 --batch_size 128 --num_encoder_layers 3 --num_decoder_layers 3 --emb_size 480 --num_ff 30 --num_epochs 100 --reg_start 10 --wl1 0.05 --logic_args 2 --block_outs 2"
SEED=1
python translation.py $PAR --seed $SEED

