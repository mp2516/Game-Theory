from mesa.visualization.ModularVisualization import VisualizationElement
from .logger import logger


def convert_polygons(polygons):
    if not polygons:
        return []
    else:
        new_polygons = []
        for rack in polygons:
            new_polygon = []
            for point in list(rack.exterior.coords):
                new_polygon.append([round(point[0], 3),
                                    round(point[1], 3)])
            new_polygons.append(new_polygon)

        return new_polygons


def convert_terminal_location(terminal_location):
    if not terminal_location:
        return []
    else:
        bounds = terminal_location.shape.bounds
        return [x for x in [bounds[0], bounds[1], bounds[2], bounds[3]]]


def convert_locations(locations):
    return [[loc.x, loc.y] for loc in locations]


# noinspection PyMethodMayBeStatic
class SimpleCanvas(VisualizationElement):
    local_includes = ['visualization/simple_continuous_canvas.js']
    portrayal_method = None

    def __init__(self, portrayal_method, shape_outline, padding=50, canvas_height=500, canvas_width=500, size=1000):
        """

        Args:
            portrayal_method: Function that accepts an object to be rendered and returns a display spec.
            shape_outline: A list of tuples defining the vertices of the floor plan.
            padding: Padding value from canvas.
            canvas_height: height of canvas to create
            canvas_width: width of canvas to create
        """
        super().__init__()
        self.portrayal_method = portrayal_method
        self.padding = padding
        self.canvas_height = canvas_height
        self.canvas_width = canvas_width
        self.half_padding = padding / 2

        shape_outline = [[point[0], point[1]] for point in shape_outline]

        canvas_config = dict(CANVAS_WIDTH=self.canvas_width,
                             CANVAS_HEIGHT=self.canvas_height,
                             OUTLINE=shape_outline,
                             PADDING=padding,
                             SIZE=size)

        new_element = ("new Simple_Continuous_Module({})".format(canvas_config))
        self.js_code = "elements.push(" + new_element + ");"

    def render(self, model):
        """
        Args:
            model: Model which contains agent_schedule to render.

        Returns:
            A list of dict containing spec for display including shape,
            colour, x, y etc.
        """
        space_state = []
        self.render_agents(model,
                           space_state,
                           route=True,
                           aim_point=True,
                           orientation=True)

        self.render_nodes(model, model.agent_schedule.agents[0], space_state)
        self.render_edges(model, model.agent_schedule.agents[0], space_state)
        return space_state

    def render_nodes(self, model, agent, space_state):
        vis_graph = model.layout.graph_map[agent.safety_radius]
        node_list = list(vis_graph.nodes)
        for num in range(vis_graph.number_of_nodes()):
            node = vis_graph.nodes[node_list[num]]
            portrayal = self.portrayal_method(node)
            portrayal["r"] = 0.1
            if node.keys():
                if node['b'] is not None:
                    if node['b'] % 2 == 0:
                        portrayal['Color'] = "Blue"
                portrayal["x"] = node['x']
                portrayal["y"] = node['y']
                portrayal["opacity"] = 0.3

            space_state.append(portrayal)

    def render_agents(self, model, space_state, route=False, aim_point=False, orientation=False):
        for agent in model.agent_schedule.agents:
            self.render_agent(agent, space_state)
            if route: self.render_route(agent, space_state)
            if aim_point: self.render_aim_point(agent, space_state)
            if orientation: self.render_orientation(agent, space_state, length=agent.safety_radius)

    def render_orientation(self, agent, space_state, length):
        portrayal = self.portrayal_method(agent.orientation)
        portrayal["x1"] = agent.position.x
        portrayal["y1"] = agent.position.y
        portrayal["x2"] = (agent.orientation.x * length) + agent.position.x
        portrayal["y2"] = (agent.orientation.y * length) + agent.position.y
        portrayal["Color"] = "black"
        portrayal["Width"] = 2
        portrayal["opacity"] = 0.7
        space_state.append(portrayal)

    def render_agent(self, agent, space_state):
        portrayal = self.portrayal_method(agent)
        x = agent.position.x
        y = agent.position.y
        portrayal["x"] = x
        portrayal["y"] = y
        portrayal["opacity"] = 1
        space_state.append(portrayal)

    def render_aim_point(self, agent, space_state):

        if agent.aim_point:
            portrayal = self.portrayal_method(agent)
            portrayal["x"] = agent.aim_point.x
            portrayal["y"] = agent.aim_point.y
            portrayal["r"] = agent.safety_radius
            portrayal["Filled"] = False
            portrayal["opacity"] = 0.7
            space_state.append(portrayal)

    def render_route(self, agent, space_state):
        if len(agent.route) > 0:
            for index in range(len(agent.route) - 1):
                portrayal = dict(Shape="line", Color=agent.color, Width=3,
                                 x1=agent.route[index].x,
                                 x2=agent.route[index + 1].x,
                                 y1=agent.route[index].y,
                                 y2=agent.route[index + 1].y,
                                 opacity=0.3)
                space_state.append(portrayal)

    def render_edges(self, model, agent, space_state):
        vis_graph = model.layout.graph_map[agent.safety_radius ]

        for edge in list(vis_graph.edges):
            node1 = vis_graph.nodes[edge[0]]
            node2 = vis_graph.nodes[edge[1]]
            # TODO: Fix generation of edges, currently produces an empty dictionary
            if node1 and node2:
                portrayal = dict(Shape="line", Color="Black", Width=2,
                                 x1=node1['x'],
                                 x2=node2['x'],
                                 y1=node1['y'],
                                 y2=node2['y'],
                                 opacity=0.1)
                space_state.append(portrayal)