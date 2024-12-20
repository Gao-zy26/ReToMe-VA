import os
CONFIG_ROOT_KINETICS = ''
CONFIG_PATH = {
    'slow_50_16': os.path.join(CONFIG_ROOT_KINETICS, 'i3d_slow_resnet50_f16s4_kinetics400.yaml'),
    'tpn_50_16': os.path.join(CONFIG_ROOT_KINETICS, 'tpn_resnet50_f16s4_kinetics400.yaml'),
    'tpn_101_16': os.path.join(CONFIG_ROOT_KINETICS, 'tpn_resnet101_f16s4_kinetics400.yaml'),
    'slow_101_16': os.path.join(CONFIG_ROOT_KINETICS, 'i3d_slow_resnet101_f16s4_kinetics400.yaml'),
    'r2plus1d_50_16': os.path.join(CONFIG_ROOT_KINETICS, 'r2plus1d_v1_resnet50_kinetics400.yaml')
    }
