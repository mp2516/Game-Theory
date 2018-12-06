var Simple_Continuous_Module = function(canvas_config) {
	/*
		canvas_width : Overall width of canvas, including padding.
		canvas_height : Overall height of canvas, including padding.
		point_list : List of points outlining warehouse.
		racks : List of [x_min, y_min, x_max, y_max] detailing racks.
	*/

	// Create the element
	// ------------------

	// Create the tag:
	canvas_height = canvas_config.CANVAS_HEIGHT;
	canvas_width = canvas_config.CANVAS_WIDTH;
	outline = canvas_config.OUTLINE;
	padding = canvas_config.PADDING;
	size = canvas_config.SIZE;

	multiplier = size / Math.min(canvas_height, canvas_width);
	canvas_height = (canvas_height * multiplier) + padding
	canvas_width = (canvas_width * multiplier) + padding;

	var canvas_tag = "<canvas width='" + canvas_width + "' height='" + canvas_height + "' ";
	canvas_tag += "style='border:1px dotted'></canvas>";
	// Append it to body:
	var canvas = $(canvas_tag)[0];
	$("body").append(canvas);

	// Create the context and the drawing controller:
	var context = canvas.getContext("2d");
	var canvasDraw = new ContinuousVisualization(canvas_height, canvas_width, outline, multiplier, padding);

	render_count = 0;

	this.render = function(data) {
		canvasDraw.resetCanvas();
		canvasDraw.draw(data);
	};

	this.reset = function() {
		canvasDraw.resetCanvas();
	};

};