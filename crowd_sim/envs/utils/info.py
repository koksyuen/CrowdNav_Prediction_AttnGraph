class Timeout(object):
    def __init__(self):
        pass

    def __str__(self):
        return 'Timeout'


class ReachGoal(object):
    def __init__(self):
        pass

    def __str__(self):
        return 'Reaching goal'


class Danger(object):
    def __init__(self, min_dist):
        self.min_dist = min_dist

    def __str__(self):
        return 'Too close: {}m'.format(self.min_dist)


class Collision(object):
    def __init__(self):
        pass

    def __str__(self):
        return 'Collision'


class OutRoad(object):
    def __init__(self):
        pass

    def __str__(self):
        return 'Out of road'


class Nothing(object):
    def __init__(self):
        pass

    def __str__(self):
        return ''


class Potential(object):
    def __init__(self, previous_potential, current_potential, reward):
        self.pre_pot = previous_potential
        self.cur_pot = current_potential
        self.reward = reward

    def __str__(self):
        return 'Previous potential: {}   Current potential: {}   Reward: {}'.format(self.pre_pot, self.cur_pot, self.reward)
