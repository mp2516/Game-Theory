//var ContinuousVisualization = function(height, width, multiplier, padding) {
//	var height = height;
//	var width = width;
//	var padding = padding/2;
//	var m = multiplier;
//
//	this.draw = function(objects, clear) {
//		for (var i in objects) {
//			var p = objects[i];
//			if (p.Shape == "rect")
//				this.drawRectangle(p.x, p.y, p.w, p.h, p.Color, p.Filled);
//			if (p.Shape == "circle")
//				console.log(p.opacity);
//				this.drawCircle(p.x, p.y, p.r, p.Color, p.Filled, p.opacity);
//			if (p.Shape == "line")
//				console.log(p.opacity);
//				this.drawLine(p.x1, p.y1, p.x2, p.y2, p.Color, p.Width, p.opacity);
//		};
//
//	};
//
//	this.drawCircle = function(x, y, radius, color, fill, opacity) {
//		var r = radius * m;
//
//		context.globalAlpha = opacity;
//
//		context.beginPath();
//		context.arc(x * m + padding, y * m + padding, r, 0, Math.PI * 2, false);
//		context.closePath();
//
//		context.strokeStyle = color;
//		context.lineWidth = 1
//		context.stroke();
//
//		if (fill) {
//			context.fillStyle = color;
//			context.fill();
//		}
//
//		context.globalAlpha = 1;
//
//	};
//
//	this.drawLine = function(x1, y1, x2, y2, color, width, opacity){
//		context.globalAlpha = opacity;
//		console.log("Drawing line..")
//		context.beginPath();
//		context.lineWidth = width;
//		context.strokeStyle = color;
//
//		context.moveTo(x1 * m + padding, y1*m + padding);
//		context.lineTo(x2*m + padding,y2*m + padding);
//		context.stroke();
//		context.globalAlpha = 1;
//	}
//
//	this.drawRectangle = function(x, y, w, h, color, fill) {
//		context.beginPath();
//		context.globalAlpha = 0.5;
//		context.strokeStyle = color;
//		context.lineWidth = 2;
//		context.fillStyle = color;
//		if (fill)
//			context.fillRect(x*m + padding, y*m + padding, w*m, h*m);
//		else
//			context.strokeRect(x*m + padding, y*m + padding, w*m, h*m);
//		context.globalAlpha = 1;
//	};
//
//	this.drawPolygon = function(point_list, stroke_color, fill_color=null, width, opacity){
//		context.globalAlpha = opacity;
//	    context.beginPath();
//	    context.lineWidth = width;
//	    context.strokeStyle = stroke_color;
//
//	    context.moveTo(point_list[0][0]*m + padding, point_list[0][1]*m + padding);
//	    for (var i = 1; i < point_list.length; i++){
//	    	context.lineTo(point_list[i][0]*m + padding, point_list[i][1]*m + padding);
//	    	context.stroke();
//	    }
//	    context.closePath();
//	    context.stroke();
//	    if (fill_color){
//	    	context.fillStyle = fill_color;
//	    	context.fill();
//		}
//		context.opacity = 1;
//	}
//
//	this.resetCanvas = function() {
//		// Draw outline of warehouse
//
//		context.clearRect(0, 0, height, width);
//		this.drawPolygon(outline, stroke_color='DarkSlateGray', fill_color='White', width=10);
//
//		context.beginPath();
//	};
//};

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