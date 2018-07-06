def add_arguments(parser):
    '''
    Add your arguments here if needed. The TAs will run test.py to load
    your default arguments.

    For example:
        parser.add_argument('--batch_size', type=int, default=32, help='batch size for training')
        parser.add_argument('--learning_rate', type=float, default=0.01, help='learning rate for training')
    '''
    BUFFER_SIZE = 10000
    EXPLORE_STEP= 3000000
    EXPLOITATION_STEP=3000000
    BATCH_SIZE =  32
    CURRENT_UPDATE_STEP = 4
    TARGET_UPDATE_STEP = 1000
    LEARNING_RATE = 1.5E-4
    GAMMA = 0.99
    EPSILON = [ 1, 0.1]
    TEST_MAX_STEP = 1000
    TENSORBOARD_DIR = 'dqn'

    parser.add_argument('--buffer_size', type=int, default=BUFFER_SIZE, help='buffer size')
    parser.add_argument('--explore_step', type=int, default=EXPLORE_STEP, help='explore step')
    parser.add_argument('--exploitation_step', type=int, default=EXPLOITATION_STEP, help='exploitation step')
    parser.add_argument('--batch_size', type=int, default=BATCH_SIZE, help='batch size')
    parser.add_argument('--current_update_step', type=int, default=CURRENT_UPDATE_STEP, help='current update step')
    parser.add_argument('--target_update_step', type=int, default=TARGET_UPDATE_STEP, help='target update step')
    parser.add_argument('--learning_rate', type=float, default=LEARNING_RATE, help='learning rate')
    parser.add_argument('--gamma', type=float, default=GAMMA, help='gamma')
    parser.add_argument('--epsilon', type=float, nargs=2, default=EPSILON, help='epsilon')
    parser.add_argument('--test_max_step', type=int, default=TEST_MAX_STEP, help='test max step')
    parser.add_argument('--tensorboard_dir', type=str, default=TENSORBOARD_DIR, help='tensorboard dir')
    return parser
