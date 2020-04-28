import gym
from gym import error, spaces, utils
from gym.utils import seeding
import numpy as np
from gurobipy import Model, quicksum, GRB
import matplotlib.pyplot as plt
import networkx as nx
import time
import pyglet
from io import BytesIO
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg


def build_base_structure(n_plant, n_product, capacity, std_mean_ratio):
    adjacency_matrix = np.zeros((n_plant, n_product))  # Adjacency Matrix

    capacity = capacity * np.ones(n_plant)  # capacity vector
    demand_mean = sum(capacity) / n_product * np.ones(n_product)  # mean demand vector
    demand_std = demand_mean * std_mean_ratio  # standard deviation vector (if needed)

    # initialize
    plant_i = 0
    product_j = 0
    sink = demand_mean[0]
    source = capacity[0]
    delta = 0.0000000001

    while plant_i < n_plant and product_j < n_product:

        # create a link between plant_i and product_j
        adjacency_matrix[plant_i, product_j] = 1

        if source > sink + delta:  # product_j demand can be met
            new_source = source - sink
            # product_j demand is met, move forward to the next product
            product_j += 1
            if product_j < n_product:
                new_sink = demand_mean[product_j]

        elif sink > source + delta:  # plant capacity can be used up

            new_sink = sink - source

            # plant capacity is used up, move forward to the next plant
            plant_i += 1
            if plant_i < n_plant:
                new_source = capacity[plant_i]

        else:
            # plant capacity matches with product demand. Move forward in both plant and product
            plant_i += 1
            product_j += 1
            if plant_i < n_plant and product_j < n_product:
                new_source = capacity[plant_i]
                new_sink = demand_mean[product_j]
        source = new_source
        sink = new_sink

    return adjacency_matrix


def expected_sales_for_structure(structure, n_sample, capacity,
                                 std_mean_ratio=None,
                                 demand_mean=None,
                                 demand_std=None,
                                 flow_profits=None,
                                 fixed_costs=None):
    # initialize for simulation and optimization
    n_plant, n_product = structure.shape
    plants = range(n_plant)
    products = range(n_product)
    samples = range(n_sample)

    # below is needed for env_version in (1, 2)
    if len(capacity) == 1:
        capacity = capacity * np.ones(n_plant)  # capacity vector
    if demand_mean is None:
        demand_mean = sum(capacity) / n_product * np.ones(n_product)  # mean demand vector
    if demand_std is None:
        demand_std = demand_mean * std_mean_ratio  # standard deviation vector (if needed)
    if flow_profits is None:
        # flow_profits = np.ones((n_plant, n_product))
        np.random.seed(3)
        flow_profits = np.random.rand(n_plant, n_product)
    if fixed_costs is None:
        fixed_costs = np.zeros((n_plant, n_product))

    # Simulate Expected Profits
    model = Model('MaxProfit')

    flow = {};
    constraint_plant = {};  # cs --> constraint of source
    constraint_product = {};  # ct --> constraint of target

    for j in products:
        for i in plants:
            # create flow variables in the model
            flow[i, j] = model.addVar(name='f_%s' % i + '%s' % j, ub=structure[i, j] * capacity[i])
            # set the coefficient of the flow variable in the objective function
            flow[i, j].setAttr(GRB.attr.Obj, flow_profits[i, j])
    model.update()

    for i in plants:
        constraint_plant[i] = model.addConstr(quicksum(flow[i, j] for j in products) <= capacity[i],
                                              name='cs_%s' % i)
    for j in products:
        constraint_product[j] = model.addConstr(quicksum(flow[i, j] for i in plants) <= 0, name='ct_%s' % j)
    model.update()

    model.setAttr("ModelSense", GRB.MAXIMIZE)
    model.setParam('OutputFlag', 0)

    # np.random.seed(0)
    sample_profits = np.ones(n_sample)

    # simulate n_sample number of demands
    # for each demand, update constraint_product RHS and run model optimization to get the max profit for the demand
    for sample in samples:
        demand = np.random.normal(demand_mean, demand_std)
        # truncate demand at two standard deviations
        demand = np.maximum(demand, demand_mean - 2 * demand_std)
        demand = np.minimum(demand, demand_mean + 2 * demand_std)
        # make sure demand is not negative
        demand = np.maximum(demand, 0.0)

        for j in products:
            constraint_product[j].setAttr(GRB.attr.RHS, demand[j])
        model.optimize()
        ## If we use sparse reward for env_version=3, then we need to substract arc costs at the end of the game
        # sample_profits[sample] = model.objVal - np.sum(np.multiply(fixed_costs, structure))

        # since a reward of negative arc cost is given at each step for env_version=3, the final reward should
        # include arc cost induced by only the last step. Note that the arc cost is not handled here, but outside this
        # calculation. Check out FlexibilityEnv.step().
        sample_profits[sample] = model.objVal

    return np.average(sample_profits), np.std(sample_profits)


def plot_structure(structure, fig=None):
    n_plant, n_product = structure.shape

    # create graph
    G = nx.Graph()
    G.add_nodes_from(["s{}".format(i) for i in range(n_plant)])
    G.add_nodes_from(["t{}".format(j) for j in range(n_product)])
    for i in range(n_plant):
        for j in range(n_product):
            if structure[i, j] == 1:
                G.add_edge('s{}'.format(i), 't{}'.format(j))

    plant_nodes = list(filter(lambda x: 's' in x, list(G.nodes)))
    product_nodes = list(filter(lambda x: 't' in x, list(G.nodes)))
    pos = _bipartite_layout(G, range(n_plant), range(n_product))

    # nx.draw_networkx(G, pos, ax=ax_outlier[i, j])
    if fig is None:
        _, ax = plt.subplots(1, 1, figsize=(12, 8))
    else:
        ax = fig.add_subplot(111)
        ax.cla()
    nx.draw_networkx_nodes(G, pos, nodelist=plant_nodes, node_size=200, node_color='green', alpha=0.2, ax=ax,
                           label='plant')
    nx.draw_networkx_nodes(G, pos, nodelist=product_nodes, node_size=200, node_color='orange', alpha=0.2, ax=ax,
                           label='product')
    nx.draw_networkx_labels(G, pos, font_color='k', font_size=10, alpha=0.8, ax=ax)
    nx.draw_networkx_edges(G, pos, edge_color='k', alpha=0.3, ax=ax)

    # plt.legend(bbox_to_anchor=(0.5,0), loc="lower right")


def _bipartite_layout(G, left_nodes, right_nodes):
    left_nodes = sorted(left_nodes, reverse=True)
    right_nodes = sorted(right_nodes, reverse=True)
    pos = {}
    length = len(left_nodes)
    delta = 1.5 / (length - 1)
    for i, node in enumerate(left_nodes):
        pos['s{}'.format(node)] = np.array([-1., -0.75 + i * delta])

    for i, node in enumerate(right_nodes):
        pos['t{}'.format(node)] = np.array([1., -0.75 + i * delta])

    return pos


def render_figure(fig):
    w, h = fig.get_size_inches()
    dpi_res = fig.get_dpi()
    w, h = int(np.ceil(w * dpi_res)), int(np.ceil(h * dpi_res))

    canvas = FigureCanvasAgg(fig)
    buffer = BytesIO()
    canvas.print_raw(buffer, dpi=dpi_res)
    return pyglet.image.ImageData(w, h, 'RGBA', buffer.getvalue(), -4 * w)


def _terminated(self):
    total_arcs_after_step = np.sum(self.adjacency_matrix)
    env1_terminated = (self.env_version == 1 and total_arcs_after_step == self.target_arcs)
    env2_terminated = (self.env_version == 2 and self.step_count == self.target_arcs)
    env345_terminated = (self.env_version > 2 and self.step_count == self.allowed_steps)
    return env1_terminated or env2_terminated or env345_terminated


def _induced_fixed_cost(self, row_index, col_index):
    if self.adjacency_matrix[row_index, col_index] == 1:
        return - self.fixed_costs[row_index, col_index]
    else:
        return self.fixed_costs[row_index, col_index]


class FlexibilityEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(
            self,
            n_plant=10,
            n_product=10,
            target_arcs=15,
            n_sample=5000,
            capacity_mean=None,
            capacity_std=None,
            reward_shaping="BASIC",
            arc_probability_numerator=0.0,
            env_version=1,
            demand_mean=None,
            demand_std=None,
            profit_matrix=None,
            fixed_costs=None,
            starting_structure=None,
            std_mean_ratio=None

    ):
        self.n_plant = n_plant
        self.n_product = n_product
        self.target_arcs = target_arcs
        self.n_sample = n_sample
        self.std_mean_ratio = std_mean_ratio
        self.adjacency_matrix = np.random.choice(np.arange(0, 2), size=(self.n_plant, self.n_product), p=[0.9, 0.1])
        self.reward_shaping = reward_shaping
        self.arc_probability_numerator = arc_probability_numerator

        # change for env2
        self.step_count = 0  # to count the number of steps that has been taken
        self.env_version = env_version

        # change for env3
        self.capacity_mean = capacity_mean
        self.capacity_std = capacity_std
        self.demand_mean = demand_mean
        self.demand_std = demand_std
        assert profit_matrix.shape == fixed_costs.shape
        assert profit_matrix.shape == starting_structure.shape
        self.profit_matrix = profit_matrix
        self.fixed_costs = fixed_costs
        self.starting_structure = starting_structure
        self.allowed_steps = int(self.target_arcs - np.sum(starting_structure))
        assert self.allowed_steps > 0, print("target_arcs = {}, sum(starting_structure) = {}"
                                             .format(self.target_arcs, np.sum(starting_structure)))

        # Env variables
        if self.env_version == 4:
            # add dummy action
            self.action_dim = n_plant * n_product + 1
        elif self.env_version > 40:
            # add dummy action
            self.action_dim = n_plant * n_product + self.env_version % 40
        else:
            self.action_dim = n_plant * n_product

        # added for env_version == 5
        self.action_is_dummy = False

        self.viewer = None
        self.state_dim = n_plant * n_product  # do not include self.target_arcs

        # Env variables needed for compatibility with gym
        self.action_space = spaces.Discrete(self.action_dim)
        self.observation_space = spaces.MultiBinary(self.state_dim)

        # added for reward_shaping SALES_INCREMENT
        if self.reward_shaping in ("SALES_INCREMENT"):
            self.expected_sales, _ = expected_sales_for_structure(self.adjacency_matrix, self.n_sample,
                                                                  self.capacity_mean,
                                                                  std_mean_ratio=self.std_mean_ratio)

    def step(self, action):
        r = 0
        done = False
        self.step_count += 1

        assert not np.sum(self.adjacency_matrix) == self.target_arcs, \
            print("in env.step(). It seems target_arcs is already met but still is stepping. adjacency_matrix: \n{}, "
                  "target_arcs: {}".format(self.adjacency_matrix, self.target_arcs))

        # assert action is in the right range
        assert 0 <= action < self.action_dim


        # action is the index of the link to be changed. If the link already exists, then the action is to remove it,
        # otherwise, the action is to add it.
        # perform adding/removing the arc identified by the action
        if self.env_version in (1, 2, 3):
            row_index = int(action / self.n_product)
            col_index = int(action % self.n_product)
            self.adjacency_matrix[row_index, col_index] = (self.adjacency_matrix[row_index, col_index] + 1) % 2
        elif self.env_version ==4 or self.env_version > 40:  # env_version == 4 or >40
            if action >= self.n_product * self.n_plant:
                # this is the dummy action. Do not change adjacency matrix.
                self.action_is_dummy = True
            else:
                self.action_is_dummy = False
                row_index = int(action / self.n_product)
                col_index = int(action % self.n_product)
                self.adjacency_matrix[row_index, col_index] = (self.adjacency_matrix[row_index, col_index] + 1) % 2
        elif self.env_version == 5:
            # Only add arcs. If an arc already exists, do nothing --> dummy action
            row_index = int(action / self.n_product)
            col_index = int(action % self.n_product)
            if self.adjacency_matrix[row_index, col_index] == 1:
                self.action_is_dummy = True
            else:
                self.action_is_dummy = False
                self.adjacency_matrix[row_index, col_index] = 1

        # reward
        if self.reward_shaping == "BASIC":
            if self.env_version in (1, 2):
                # there is no fixed arc cost for env_version 1 and 2
                r = 0
            if self.env_version == 3:
                # reward is the induced fixed cost by adding/removing an arc for each step before termination
                r = _induced_fixed_cost(self, row_index, col_index)
            if self.env_version in (4, 5) or self.env_version > 40:
                if self.action_is_dummy:
                    # dummy action. there is no fixed cost induced.
                    r = 0
                else:  # non-dummy-action
                    # reward is the induced fixed cost by adding/removing an arc for each step before termination
                    # note that in enn version 5, arcs are only added
                    r = _induced_fixed_cost(self, row_index, col_index)

            if _terminated(self):
                done = True
                if self.env_version in (1, 2):
                    # evaluate structure performance
                    structure_performance, _ = expected_sales_for_structure(self.adjacency_matrix,
                                                                            self.n_sample,
                                                                            self.capacity_mean,
                                                                            std_mean_ratio=self.std_mean_ratio)

                    r += structure_performance

                else:  # self.env_version >= 3:
                    # evaluate structure performance
                    structure_performance, _ = expected_sales_for_structure(self.adjacency_matrix,
                                                                            self.n_sample,
                                                                            self.capacity_mean,
                                                                            demand_mean=self.demand_mean,
                                                                            demand_std=self.demand_std,
                                                                            flow_profits=self.profit_matrix,
                                                                            fixed_costs=self.fixed_costs)
                    r += structure_performance

        elif self.reward_shaping == "SALES_INCREMENT":
            # evaluate structure performance
            sales, _ = expected_sales_for_structure(self.adjacency_matrix, self.n_sample, self.capacity_mean,
                                                    std_mean_ratio=self.std_mean_ratio)
            r = sales - self.expected_sales
            self.expected_sales = sales

            # done?
            if _terminated(self):
                done = True

        s_ = np.copy(np.squeeze(self.adjacency_matrix.reshape(1, -1)))

        return s_, r, done, {}

    def reset(self):
        # reset adjacency_matrix
        if self.env_version in (1, 2):
            self.adjacency_matrix = np.random.choice(2, size=(self.n_plant, self.n_product),
                                                     p=[1.0 - self.arc_probability_numerator / self.n_plant,
                                                        self.arc_probability_numerator / self.n_plant])
        else:  # env_version in (3, 4, >40, 5)
            self.adjacency_matrix = np.copy(self.starting_structure)

        # reset step count
        self.step_count = 0

        if self.reward_shaping in ("SALES_INCREMENT", "VR"):
            self.expected_sales, _ = expected_sales_for_structure(self.adjacency_matrix, self.n_sample,
                                                                  self.capacity_mean,
                                                                  std_mean_ratio=self.std_mean_ratio)
            # print("in env.reset(), A: {}".format(self.adjacency_matrix))
            # print("in env.reset(), expected sales: {}".format(self.expected_sales))

        s = np.copy(np.squeeze(self.adjacency_matrix.reshape(1, -1)))
        return s

    def render(self):
        if self.viewer is None:
            self.viewer = Viewer(self.adjacency_matrix
                                 , width=800, height=500
                                 )
        self.viewer.adjacency_matrix = self.adjacency_matrix
        self.viewer.render()

    def sample_action(self):
        return np.random.randint(0, self.n_plant * self.n_product)

    def close(self):
        if self.viewer is not None:
            self.viewer.close()


class Viewer(pyglet.window.Window):

    def __init__(self, adjacency_matrix, width=600, height=400):
        self.adjacency_matrix = adjacency_matrix

        vsync = False  # to not use the monitor FPS, we can speed up training
        super(Viewer, self).__init__(width=width, height=height, resizable=False, caption='Flexibility', vsync=vsync)
        dpi_res = min(self.width, self.height) / 4 / np.floor(height / 200)
        self.fig = Figure((self.width / dpi_res, self.height / dpi_res), dpi=dpi_res)
        self.image = None

    def render(self):
        self._update_image()
        self.switch_to()
        self.dispatch_events()
        self.dispatch_event('on_draw')
        self.flip()

    def on_draw(self):
        self.clear()
        self.image.blit(0, 0)

    def _update_image(self):
        plot_structure(self.adjacency_matrix, self.fig)
        # self.image = self._render_figure(fig)
        w, h = self.fig.get_size_inches()
        dpi_res = self.fig.get_dpi()
        w, h = int(np.ceil(w * dpi_res)), int(np.ceil(h * dpi_res))

        canvas = FigureCanvasAgg(self.fig)
        buffer = BytesIO()
        canvas.print_raw(buffer, dpi=dpi_res)
        self.image = pyglet.image.ImageData(w, h, 'RGBA', buffer.getvalue(), -4 * w)


if __name__ == '__main__':
    ### calculate expected sales of structure with full flexibility 10x10
    structure = np.ones((10, 10))
    n_sample, capacity, std_mean_ratio = 20000, 100.0, 0.4
    mean, std = expected_sales_for_structure(structure, n_sample, capacity, std_mean_ratio=std_mean_ratio)
    print("Expected sales for 10x10 full flexibility with capacity {}, n_sample {}, and std_mean_ratio {} is: {} "
          "with std {}"
          .format(capacity, n_sample, std_mean_ratio, mean, std))

    ### calculate expected sales of structure with full flexibility 20x20
    structure = np.ones((20, 20))
    n_sample, capacity, std_mean_ratio = 20000, 100.0, 0.4
    mean, std = expected_sales_for_structure(structure, n_sample, capacity, std_mean_ratio=std_mean_ratio)
    print("Expected sales for 20x20 full flexibility with capacity {}, n_sample {}, and std_mean_ratio {} is: {} "
          "with std {}"
          .format(capacity, n_sample, std_mean_ratio, mean, std))

    # ### testing whether env.close() works
    # env = FlexibilityEnv(n_plant=10, n_product=10, target_arcs=20)
    # # env.render()
    # count = 0
    # print("render 10x10")
    # env.render()
    # env.render()
    # print("wait for 5 seconds")
    #
    # time.sleep(5)
    # env.close()
    # print("10x10 closed")
    #
    # print("render 20x20")
    # env = FlexibilityEnv(n_plant=20, n_product=20, target_arcs=20)
    # env.render()
    # env.render()
    # print("wait for 5 seconds")
    # time.sleep(5)
    # env.close()
    # print("20x20 closed")
