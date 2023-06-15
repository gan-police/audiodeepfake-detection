#!/bin/bash

source ${HOME}/.bashrc

# WPT 256
sbatch ./scripts/eval.sh "./exp/log5/models/fake_packetssym8_none_100_22050_22050_256_1-11025_0.7_0.0001_0.01_128_2_10e_lcnn_signsFalse_augcFalse_augnFalse_power1.0" packets 256 sym8 1.0 False
sbatch ./scripts/eval.sh "./exp/log5/models/fake_packetssym8_none_100_22050_22050_256_1-11025_0.7_0.0001_0.01_128_2_10e_lcnn_signsFalse_augcFalse_augnFalse_power2.0" packets 256 sym8 2.0 False
sbatch ./scripts/eval.sh "./exp/log5/models/fake_packetssym8_none_100_22050_22050_256_1-11025_0.7_0.0001_0.01_128_2_10e_lcnn_signsTrue_augcFalse_augnFalse_power1.0" packets 256 sym8 1.0 True
sbatch ./scripts/eval.sh "./exp/log5/models/fake_packetssym8_none_100_22050_22050_256_1-11025_0.7_0.0001_0.01_128_2_10e_lcnn_signsTrue_augcFalse_augnFalse_power2.0" packets 256 sym8 2.0 True
sbatch ./scripts/eval.sh "./exp/log5/models/fake_packetsdb8_none_100_22050_22050_256_1-11025_0.7_0.0001_0.01_128_2_10e_lcnn_signsFalse_augcFalse_augnFalse_power1.0" packets 256 db8 1.0 False
sbatch ./scripts/eval.sh "./exp/log5/models/fake_packetsdb8_none_100_22050_22050_256_1-11025_0.7_0.0001_0.01_128_2_10e_lcnn_signsFalse_augcFalse_augnFalse_power2.0" packets 256 db8 2.0 False
sbatch ./scripts/eval.sh "./exp/log5/models/fake_packetsdb8_none_100_22050_22050_256_1-11025_0.7_0.0001_0.01_128_2_10e_lcnn_signsTrue_augcFalse_augnFalse_power1.0" packets 256 db8 1.0 True
sbatch ./scripts/eval.sh "./exp/log5/models/fake_packetsdb8_none_100_22050_22050_256_1-11025_0.7_0.0001_0.01_128_2_10e_lcnn_signsTrue_augcFalse_augnFalse_power2.0" packets 256 db8 2.0 True
sbatch ./scripts/eval.sh "./exp/log5/models/fake_packetssym20_none_100_22050_22050_256_1-11025_0.7_0.0001_0.01_128_2_10e_lcnn_signsFalse_augcFalse_augnFalse_power1.0" packets 256 sym20 1.0 False
sbatch ./scripts/eval.sh "./exp/log5/models/fake_packetssym20_none_100_22050_22050_256_1-11025_0.7_0.0001_0.01_128_2_10e_lcnn_signsFalse_augcFalse_augnFalse_power2.0" packets 256 sym20 2.0 False
sbatch ./scripts/eval.sh "./exp/log5/models/fake_packetssym20_none_100_22050_22050_256_1-11025_0.7_0.0001_0.01_128_2_10e_lcnn_signsTrue_augcFalse_augnFalse_power1.0" packets 256 sym20 1.0 True
sbatch ./scripts/eval.sh "./exp/log5/models/fake_packetssym20_none_100_22050_22050_256_1-11025_0.7_0.0001_0.01_128_2_10e_lcnn_signsTrue_augcFalse_augnFalse_power2.0" packets 256 sym20 2.0 True

# new
sbatch ./scripts/eval.sh "./exp/log5/models/fake_packetshaar_none_100_22050_22050_256_1-11025_0.7_0.0001_0.01_128_2_10e_lcnn_signsFalse_augcFalse_augnFalse_power1.0" packets 256 haar 1.0 False
sbatch ./scripts/eval.sh "./exp/log5/models/fake_packetshaar_none_100_22050_22050_256_1-11025_0.7_0.0001_0.01_128_2_10e_lcnn_signsFalse_augcFalse_augnFalse_power2.0" packets 256 haar 2.0 False
sbatch ./scripts/eval.sh "./exp/log5/models/fake_packetshaar_none_100_22050_22050_256_1-11025_0.7_0.0001_0.01_128_2_10e_lcnn_signsTrue_augcFalse_augnFalse_power1.0" packets 256 haar 1.0 True
sbatch ./scripts/eval.sh "./exp/log5/models/fake_packetshaar_none_100_22050_22050_256_1-11025_0.7_0.0001_0.01_128_2_10e_lcnn_signsTrue_augcFalse_augnFalse_power2.0" packets 256 haar 2.0 True
# new end

# STFT 256
sbatch ./scripts/eval.sh "./exp/log5/models/fake_stft_none_100_22050_22050_256_1-11025_0.7_0.0001_0.01_128_2_10e_lcnn_signsFalse_augcFalse_augnFalse_power1.0" stft 256 none 1.0 False
sbatch ./scripts/eval.sh "./exp/log5/models/fake_stft_none_100_22050_22050_256_1-11025_0.7_0.0001_0.01_128_2_10e_lcnn_signsFalse_augcFalse_augnFalse_power2.0" stft 256 none 2.0 False

# WPT 512
sbatch ./scripts/eval.sh "./exp/log5/models/fake_packetssym8_none_100_22050_22050_512_1-11025_0.7_0.0001_0.01_128_2_10e_lcnn_signsFalse_augcFalse_augnFalse_power1.0" packets 512 sym8 1.0 False
sbatch ./scripts/eval.sh "./exp/log5/models/fake_packetssym8_none_100_22050_22050_512_1-11025_0.7_0.0001_0.01_128_2_10e_lcnn_signsFalse_augcFalse_augnFalse_power2.0" packets 512 sym8 2.0 False
sbatch ./scripts/eval.sh "./exp/log5/models/fake_packetsdb8_none_100_22050_22050_512_1-11025_0.7_0.0001_0.01_128_2_10e_lcnn_signsFalse_augcFalse_augnFalse_power1.0" packets 512 db8 1.0 False
sbatch ./scripts/eval.sh "./exp/log5/models/fake_packetsdb8_none_100_22050_22050_512_1-11025_0.7_0.0001_0.01_128_2_10e_lcnn_signsFalse_augcFalse_augnFalse_power2.0" packets 512 db8 2.0 False
sbatch ./scripts/eval.sh "./exp/log5/models/fake_packetssym20_none_100_22050_22050_512_1-11025_0.7_0.0001_0.01_128_2_10e_lcnn_signsFalse_augcFalse_augnFalse_power1.0" packets 512 sym20 1.0 False

# STFT 512
sbatch ./scripts/eval.sh "./exp/log5/models/fake_stft_none_100_22050_22050_512_1-11025_0.7_0.0001_0.01_128_2_10e_lcnn_signsFalse_augcFalse_augnFalse_power1.0" stft 512 none 1.0 False
sbatch ./scripts/eval.sh "./exp/log5/models/fake_stft_none_100_22050_22050_512_1-11025_0.7_0.0001_0.01_128_2_10e_lcnn_signsFalse_augcFalse_augnFalse_power2.0" stft 512 none 2.0 False
