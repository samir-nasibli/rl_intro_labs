import torch
import torch.nn as nn
import torch.nn.functional as F


class ActorCriticModel(nn.Module):

    def __init__(self, n_input, n_output, n_hidden):
        super(ActorCriticModel, self).__init__()
        self.fc = nn.Linear(n_input, n_hidden)
        self.mu = nn.Linear(n_hidden, n_output)
        self.sigma = nn.Linear(n_hidden, n_output)
        self.value = nn.Linear(n_hidden, 1)
        self.distribution = torch.distributions.Normal

    def forward(self, x):
        x = F.relu(self.fc(x))
        mu = 2 * torch.tanh(self.mu(x))
        sigma = F.softplus(self.sigma(x)) + 1e-5
        dist = self.distribution(mu.view(1, ).data, sigma.view(1, ).data)
        value = self.value(x)
        return dist, value


class PolicyNetwork():

    def __init__(self, n_state, n_action, n_hidden=50, lr=0.001):
        self.model = ActorCriticModel(n_state, n_action, n_hidden)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr)

    def predict(self, s):
        """
        Compute the output using the continuous Actor Critic model
        @param s: input state
        @return: Gaussian distribution, state_value
        """
        self.model.training = False
        return self.model(torch.Tensor(s))

    def get_action(self, s):
        """
        Estimate the policy and sample an action,
        compute its log probability
        @param s: input state
        @return: the selected action, log probability,
        predicted state-value
        """
        dist, state_value = self.predict(s)
        action = dist.sample().numpy()
        log_prob = dist.log_prob(action[0])
        return action, log_prob, state_value

    def update(self, returns, log_probs, state_values):
        """
        Update the weights of the Actor Critic network given the training samples
        @param returns: return (cumulative rewards) for each step in an episode
        @param log_probs: log probability for each step
        @param state_values: state-value for each step

        """
        loss = 0
        for log_prob, value, Gt in zip(log_probs, state_values, returns):
            advantage = Gt - value.item()
            policy_loss = -log_prob * advantage
            value_resh = torch.tensor(value.item(), requires_grad=True)
            value_loss = F.smooth_l1_loss(value_resh, Gt)
            loss += policy_loss + value_loss
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
