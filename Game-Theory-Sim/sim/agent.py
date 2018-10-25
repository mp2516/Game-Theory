from mesa import Agent
import random


class Agent(Agent):
    def __init__(self, unique_id, model):
        super().__init__(unique_id, model)
        self.value = random.randint(1,3)
        self.success = 0
        """
        Value explanation:
        1 - paper (red), 2 - rock (grey), 3 - scissors (cyan)
            1 beats 2 but loses to 3
            2 beats 3 but loses to 1
            3 beats 1 but loses to 2
        """


    def move(self):
        possible_steps = self.model.grid.get_neighborhood(
            self.pos,
            moore=True,
            include_center=False)
        new_position = random.choice(possible_steps)
        self.model.grid.move_agent(self, new_position)

    def rock_paper_scissors(self):
        neighbours = self.model.grid.get_neighbors(
            pos=self.pos,
            moore=False,
            include_center=False)
        for neighbour in neighbours:
            if self.value == 1 and neighbour.value == 3:
                self.success += 1
            elif self.value == 2 and neighbour.value == 1:
                self.value = 1
            elif self.value == 3 and neighbour.value == 2:
                self.value = 2



    def step(self):
        # self.move()
        self.rock_paper_scissors()
