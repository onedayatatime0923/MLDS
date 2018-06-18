def add_arguments(parser):
    '''
    Add your arguments here if needed. The TAs will run test.py to load
    your default arguments.

    For example:
        parser.add_argument('--batch_size', type=int, default=32, help='batch size for training')
        parser.add_argument('--learning_rate', type=float, default=0.01, help='learning rate for training')
    '''
    parser.add_argument('--buffer_size', type=int, default=10000, help='buffer size for training')
    parser.add_argument('--current_update_step', type=int, default=4, help='current update step for training')
    parser.add_argument('--target_update_step', type=int, default=1000, help='target update step for training')

    parser.add_argument('--step_n', type=int, default=50000000, help='buffer size for training')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size for training')
    parser.add_argument('--learning_rate', type=float, default=1.5E-4, help='learning rate for training')
    parser.add_argument('--epsilon', type=float, nargs=2, default=[1, 0.025], help='epsilon for training')
    return parser
