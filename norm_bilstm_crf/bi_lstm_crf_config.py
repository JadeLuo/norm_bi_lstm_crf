#encoding=utf-8
class Config():
    def __init__(self):
        self.init_learning_rate=0.01
        self.batch_size=64
        self.num_epochs=2
        self.drop_out=0.9
        self.clip_grad=5.0
        self.lstm_hidden_size=300
        self.embedding_size=300
        self.num_tags=5
        self.save_per_batch=100
        self.print_per_batch=10