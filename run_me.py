from functions.main import Main


def set_values():
    learning_rate = 0.001
    training_iters = 250000
    batch_size = 512
    display_step = 10
    n_input = 784
    n_classes = 10
    dropout = 0.75
    Main(learning_rate, training_iters, batch_size, display_step, n_input, n_classes, dropout).main()

if __name__ == '__main__':
    set_values()
