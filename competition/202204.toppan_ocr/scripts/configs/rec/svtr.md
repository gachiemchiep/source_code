english


SVTR Tiny

"Architecture:
  model_type: rec
  algorithm: SVTR
  Transform:
    name: STN_ON
    tps_inputsize: [32, 64]
    tps_outputsize: [32, 100]
    num_control_points: 20
    tps_margins: [0.05,0.05]
    stn_activation: none
  Backbone:
    name: SVTRNet
    img_size: [32, 100]
    out_char_num: 25
    out_channels: 192
    patch_merging: 'Conv'
    embed_dim: [64, 128, 256]
    depth: [3, 6, 3]
    num_heads: [2, 4, 8]
    mixer: ['Local','Local','Local','Local','Local','Local','Global','Global','Global','Global','Global','Global']
    local_mixer: [[7, 11], [7, 11], [7, 11]]
    last_stage: True
    prenorm: false
  Neck:
    name: SequenceEncoder
    encoder_type: reshape
  Head:
    name: CTCHead

      - RecResizeImg:
          character_dict_path:
          image_shape: [3, 64, 256]
          padding: False"


SVTR Small

"Architecture:
  model_type: rec
  algorithm: SVTR
  Transform:
    name: STN_ON
    tps_inputsize: [32, 64]
    tps_outputsize: [32, 100]
    num_control_points: 20
    tps_margins: [0.05,0.05]
    stn_activation: none
  Backbone:
    name: SVTRNet
    img_size: [32, 100]
    out_char_num: 25
    out_channels: 192
    patch_merging: 'Conv'
    embed_dim: [96, 192, 256]
    depth: [3, 6, 6]
    num_heads: [3, 6, 8]
    mixer: ['Local','Local','Local','Local','Local','Local','Local','Local','Global','Global','Global','Global','Global','Global','Global']
    local_mixer: [[7, 11], [7, 11], [7, 11]]
    last_stage: True
    prenorm: False
  Neck:
    name: SequenceEncoder
    encoder_type: reshape
  Head:
    name: CTCHead"

"      - RecResizeImg:
          character_dict_path:
          image_shape: [3, 64, 256]
          padding: False"    


SVTR Base

"Architecture:
  model_type: rec
  algorithm: SVTR
  Transform:
    name: STN_ON
    tps_inputsize: [32, 64]
    tps_outputsize: [48, 160]
    num_control_points: 20
    tps_margins: [0.05,0.05]
    stn_activation: none
  Backbone:
    name: SVTRNet
    img_size: [48, 160]
    out_char_num: 40     # output char patch
    out_channels: 256    # char patch dim
    patch_merging: 'Conv'
    embed_dim: [128, 256, 384]
    depth: [3, 6, 9]
    num_heads: [4, 8, 12]
    mixer: ['Local','Local','Local','Local','Local','Local','Local','Local','Global','Global','Global','Global','Global','Global','Global','Global','Global','Global']
    local_mixer: [[7, 11], [7, 11], [7, 11]]
    last_stage: True
    prenorm: False
  Neck:
    name: SequenceEncoder
    encoder_type: reshape
  Head:
    name: CTCHead"

"      - RecResizeImg:
          character_dict_path:
          image_shape: [3, 64, 256]
          padding: False"


SVTR Large

"Architecture:
  model_type: rec
  algorithm: SVTR
  Transform:
    name: STN_ON
    tps_inputsize: [32, 64]
    tps_outputsize: [48, 160]
    num_control_points: 20
    tps_margins: [0.05,0.05]
    stn_activation: none
  Backbone:
    name: SVTRNet
    img_size: [48, 160]
    out_char_num: 40
    out_channels: 384
    patch_merging: 'Conv'
    embed_dim: [192, 256, 512]
    depth: [3, 9, 9]
    num_heads: [6, 8, 16]
    mixer: ['Local','Local','Local','Local','Local','Local','Local','Local','Local','Local','Global','Global','Global','Global','Global','Global','Global','Global','Global','Global','Global']
    local_mixer: [[7, 11], [7, 11], [7, 11]]
    prenorm: false
  Neck:
    name: SequenceEncoder
    encoder_type: reshape
  Head:
    name: CTCHead"


"      - RecResizeImg:
          character_dict_path:
          image_shape: [3, 64, 256]
          padding: False"


Chinese

SVTR Tiny

Architecture:
  model_type: rec
  algorithm: SVTR
  Transform:
    name: STN_ON
    tps_inputsize: [32, 64]
    tps_outputsize: [32, 320]
    num_control_points: 20
    tps_margins: [0.05,0.05]
    stn_activation: none
  Backbone:
    name: SVTRNet
    img_size: [32, 320]
    out_char_num: 40
    out_channels: 192
    patch_merging: 'Conv'
    embed_dim: [64, 128, 256]
    depth: [3, 6, 3]
    num_heads: [2, 4, 8]
    mixer: ['Local','Local','Local','Local','Local','Local','Global','Global','Global','Global','Global','Global']
    local_mixer: [[7, 11], [7, 11], [7, 11]]
    last_stage: True
    prenorm: false
  Neck:
    name: SequenceEncoder
    encoder_type: reshape
  Head:
    name: CTCHead

      - RecResizeImg:
          image_shape: [3, 64, 256]
          padding: False
      - KeepKeys:
          keep_keys: ['image', 'label', 'length'] # dataloader will return list in this order


Small

Architecture:
  model_type: rec
  algorithm: SVTR
  Transform:
    name: STN_ON
    tps_inputsize: [32, 64]
    tps_outputsize: [32, 320]
    num_control_points: 20
    tps_margins: [0.05,0.05]
    stn_activation: none
  Backbone:
    name: SVTRNet
    img_size: [32, 320]
    out_char_num: 40
    out_channels: 192
    patch_merging: 'Conv'
    embed_dim: [96, 192, 256]
    depth: [3, 6, 6]
    num_heads: [3, 6, 8]
    mixer: ['Local','Local','Local','Local','Local','Local','Local','Local','Global','Global','Global','Global','Global','Global','Global']
    local_mixer: [[7, 11], [7, 11], [7, 11]]
    last_stage: True
    prenorm: False
  Neck:
    name: SequenceEncoder
    encoder_type: reshape
  Head:
    name: CTCHead

          - RecResizeImg:
          image_shape: [3, 64, 256]
          padding: False


Large


Architecture:
  model_type: rec
  algorithm: SVTR
  Transform:
    name: STN_ON
    tps_inputsize: [32, 64]
    tps_outputsize: [32, 320]
    num_control_points: 20
    tps_margins: [0.05,0.05]
    stn_activation: none
  Backbone:
    name: SVTRNet
    img_size: [32, 320]
    out_char_num: 40
    out_channels: 384
    patch_merging: 'Conv'
    embed_dim: [192, 256, 512]
    depth: [3, 9, 9]
    num_heads: [6, 8, 16]
    mixer: ['Local','Local','Local','Local','Local','Local','Local','Local','Local','Local','Global','Global','Global','Global','Global','Global','Global','Global','Global','Global','Global']
    local_mixer: [[7, 11], [7, 11], [7, 11]]
    prenorm: False
  Neck:
    name: SequenceEncoder
    encoder_type: reshape
  Head:
    name: CTCHead


      - RecResizeImg:
          image_shape: [3, 64, 256]
          padding: False