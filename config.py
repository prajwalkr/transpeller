import configargparse

pad_token = '§'

def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def load_args(default_config=None):

  parser = configargparse.ArgumentParser(description = "main")
  parser.add_argument('--device', type=str, default='cuda')
  parser.add_argument('--builder', default='localizer_ctc')
  parser.add_argument('--ckpt_path', default=None)
  
  # Data loader parameters
  parser.add_argument('--feat_root', type=str, required=True)
  parser.add_argument('--vocab_file', type=str, default='data/vocab.txt', 
                    help='Path to character-level vocab')
  parser.add_argument('--test_csv', type=str, required=True, 
                    help='Path to test csv')
  parser.add_argument('--bs', type=int, default=32)
  parser.add_argument('--num_workers', type=int, default=8)
  parser.add_argument('--upsampling', type=int, default=1)
  parser.add_argument('--downsampling', type=int, default=4)
  parser.add_argument('--full_word_test', action="store_true")

  # Transformer config
  parser.add_argument('--feat_dim', type=int, default=768, help='video feature dimension')
  parser.add_argument('--num_blocks', type=int, default=6, help='# of transformer blocks')
  parser.add_argument('--hidden_units', type=int, default=512, help='Transformer model size')
  parser.add_argument('--num_heads', type=int, default=8, help='# of attention heads')
  parser.add_argument('--dropout_rate', type=float, default=0.1, help='Dropout probability')

  # inference params
  parser.add_argument('--beam', type=int, default=30)
  parser.add_argument('--frame_stride', type=int, default=4, help='features frames stride')

  args = parser.parse_args()

  return args

