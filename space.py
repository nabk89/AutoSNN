
""" micro search space
"""
CANDIDATE_BLOCKS = [
  'skip_connect',
  'SCB_k3',
  'SCB_k5',
  'SRB_k3',
  'SRB_k5',
]


""" macro search space 
"""
## AutoSNN_16 = SNN_1
AutoSNN_16 = {
  'stem_channel': 16,
  'block_channels': [16,
                     'm', 32, 32,
                     'm', 64, 64,
                     'm',
                    ],
  'strides': [1, 
              2, 1, 1,
              2, 1, 1,
              2,
             ],
  'use_GAP': False,
}

AutoSNN_32 = {
  'stem_channel': 32,
  'block_channels': [32,
                     'm', 64, 64,
                     'm', 128, 128,
                     'm',
                    ],
  'strides': [1, 
              2, 1, 1,
              2, 1, 1,
              2,
             ],
  'use_GAP': False,
}

AutoSNN_64 = {
  'stem_channel': 64,
  'block_channels': [64,
                     'm', 128, 128,
                     'm', 256, 256,
                     'm',
                    ],
  'strides': [1, 
              2, 1, 1,
              2, 1, 1,
              2,
             ],
  'use_GAP': False,
}

AutoSNN_128 = {
  'stem_channel': 128,
  'block_channels': [128,
                     'm', 256, 256,
                     'm', 512, 512,
                     'm',
                    ],
  'strides': [1, 
              2, 1, 1,
              2, 1, 1,
              2,
             ],
  'use_GAP': False,
}

SNN_2 = {
  'stem_channel': 16,
  'block_channels': [16,
                     'm', 32, 32,
                     'm', 64, 64,
                     'm',
                    ],
  'strides': [1, 
              2, 1, 1,
              2, 1, 1,
              2,
             ],
  'use_GAP': True,
}

SNN_3 = {
  'stem_channel': 16,
  'block_channels': [16,
                     32, 32, 32,
                     64, 64, 64,
                     128,
                    ],
  'strides': [1, 
              2, 1, 1,
              2, 1, 1,
              2,
             ],
  'use_GAP': False,
}

SNN_4 = {
  'stem_channel': 16,
  'block_channels': [16,
                     'a', 32, 32,
                     'a', 64, 64,
                     'a',
                    ],
  'strides': [1, 
              2, 1, 1,
              2, 1, 1,
              2,
             ],
  'use_GAP': False,
}

MACRO_SEARCH_SPACE = {
    'AutoSNN_16': AutoSNN_16,
    'AutoSNN_32': AutoSNN_32,
    'AutoSNN_64': AutoSNN_64,
    'AutoSNN_128': AutoSNN_128,
    'SNN_2': SNN_2,
    'SNN_3': SNN_3,
    'SNN_4': SNN_4,
}
