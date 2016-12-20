import theano
import theano.tensor as T
from theano import printing
import numpy as np
import gym
import cPickle

rng = np.random.RandomState(23455)
input_dim = 80*80
hidden_dim = 200
gamma = 0.99
learning_rate = 1e-4
decay_rate = 0.99 # decay factor for RMSProp leaky sum of grad^2


class PolicyGradient(object):
    def __init__(self, init_file=None):
        w_bound = np.sqrt(80*80)
        W1 = np.asarray( rng.uniform(
                low=-1.0 / w_bound,
                high=1.0 / w_bound,
                size=(input_dim, hidden_dim)),
                dtype=theano.config.floatX)
        W2 = np.asarray( rng.uniform( low=-1.0 / w_bound, high=-1.0 / w_bound,
            size=(hidden_dim, 1)), dtype=theano.config.floatX)
        print(W1)
        print(W2)
        if init_file is not None:
            W3, W4 = self.load_params(init_file)
            print(W3)
            print(W4)
        self.W1 = theano.shared(
                value=W1,
                name="W1", borrow=True
                )
       # self.b1 = theano.shared(
       #         value=np.zeros(hidden_dim, dtype=theano.config.floatX),
       #         name="b1", borrow=True
       #         )
        self.W2 = theano.shared(
                value=W2,
                name="W2", borrow=True
                )
       # self.b2 = theano.shared(
       #         value=np.zeros(1, dtype=theano.config.floatX),
       #         name="b2", borrow=True
       #         )

        #self.params = [self.W1, self.b1, self.W2, self.b2]
        self.params = [self.W1, self.W2]
        self.x = T.dvector("x")

    def load_params(self, init_file):
        return cPickle.load(open(init_file, 'rb'))

    def get_hidden_values(self, x):
        return T.nnet.relu(T.dot(x, self.W1))# + self.b1)

    def get_output_values(self, x):
        logP = T.dot(x, self.W2)# + self.b2
        return logP

#    def get_cost_updates(self):
#        z = get_hidden_values(self.x)
#        dlogP, y = get_output_values(z)
#
#        loss = sum(reward*)
#
    def RMSprop(self, cost, params, lr=0.001, rho=0.9, epsilon=1e-6):
        grads = T.grad(cost=cost, wrt=params)
        updates = []
        for p, g in zip(params, grads):
            acc = theano.shared(p.get_value() * 0.)
            acc_new = rho * acc + (1 - rho) * g ** 2
            gradient_scaling = T.sqrt(acc_new + epsilon)
            g = g / gradient_scaling
            updates.append((acc, acc_new))
            updates.append((p, p - lr * g))
        return updates

    def print_weights(self):
        print("First layer weights:")
        print(self.W1)
        print("Second layer weights")
        print(self.W2)

    def get_train_fn(self):
        X = T.dmatrix("X")
        Y = self.get_output_values(self.get_hidden_values(X))
        self.f1 =  theano.function(
                [X], Y
                )

        advantage = T.dmatrix("advantage")
        loss = -T.sum(advantage*Y)
        #updates = self.get_cost_updates(X, loss)

        grad_params = T.grad(loss, self.params)
        param_printing_op = printing.Print("Param")
        param_printing = param_printing_op(grad_params[0])
        updates = [
                (param, param - learning_rate*grad_param) for param, grad_param in zip(self.params, grad_params)
                ]
        #updates = self.RMSprop(loss, self.params)

        self.f2 = theano.function(
                [X, advantage], [param_printing], updates=updates
                )



def prepro(I):
  """ prepro 210x160x3 uint8 frame into 6400 (80x80) 1D float vector """
  I = I[35:195] # crop
  I = I[::2,::2,0] # downsample by factor of 2
  I[I == 144] = 0 # erase background (background type 1)
  I[I == 109] = 0 # erase background (background type 2)
  I[I != 0] = 1 # everything else (paddles, ball) just set to 1
  return I.astype(np.float).ravel()

def init_env():
    env = gym.make('Pong-v0')
    return env

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def discount_rewards(r):
    """ take 1D float array of rewards and compute discounted reward """
    discounted_r = np.zeros_like(r)
    running_add = 0
    print(np.sum(np.asarray(r)))
    for t in reversed(xrange(0, r.size)):
        if r[t] != 0:
            running_add = 0 # reset the sum, since this was a game boundary (pong specific!)
        running_add = running_add * gamma + r[t]
        discounted_r[t] = running_add
    return discounted_r


def main():
    env = init_env()
    obs = env.reset()
    prev_x = None
    reward_sum = 0
    episode_num = 0

    #pg = PolicyGradient("models/simple_model.save")
    pg = PolicyGradient()

    num_games = 100

    pg.get_train_fn()
    for game in range(num_games):
        print("Game number: %d" % game)
        if game % 5 == 0:
            f = file('models/simple_model.save', 'wb')
            cPickle.dump([param.get_value() for param in pg.params], f, protocol=cPickle.HIGHEST_PROTOCOL)

        Action, Reward, X = [], [], []
        obs = env.reset()
        prev_x = None
        done = 0
        while not done:
            env.render()
            # Preprocessing the input
            cur_x = prepro(obs)
            x = cur_x - prev_x if prev_x is not None else np.zeros(input_dim)
            prev_x = cur_x

            X.append(x)

            ret = pg.f1(x.reshape((1, input_dim)))[0]
            aprob = sigmoid(ret)
            action = 2 if np.random.uniform() < aprob else 3 # roll the dice!

            obs, reward, done, info = env.step(action)
            Reward.append(reward)
        Reward = np.asarray(Reward)
        discounted_rewards = discount_rewards(Reward)
        discounted_rewards -= np.mean(discounted_rewards)
        discounted_rewards /= np.std(discounted_rewards)
        #print(discounted_rewards)

        pg.f2(X, discounted_rewards.reshape((len(discounted_rewards), 1)))


main()
