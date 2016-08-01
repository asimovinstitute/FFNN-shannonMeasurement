var iterations = 1;
var cells = [];
var momentum = 0;
var sampleData = [];
var learningRate = 0;

function buildBrain (layers) {
	
	for (var a = 0; a < layers.length - 1; a++) layers[a]++;
	
	cells[0] = [];
	
	for (var a = 0; a < layers[0]; a++) {
		
		cells[0][a] = {value:0, bias:a == layers[0] - 1};
		
		if (cells[0][a].bias) cells[0][a].value = -1;
		
	}
	
	for (var a = 1; a < layers.length; a++) {
		
		cells[a] = [];
		
		for (var b = 0; b < layers[a]; b++) {
			
			cells[a][b] = {error:0, value:0, weights:[], lastWeights:[], bias:false};
			
			if (a < layers.length - 1 && b == layers[a] - 1) {
				
				cells[a][b].bias = true;
				cells[a][b].value = -1;
				
			}
			
			for (var c = 0; c < layers[a - 1]; c++) {
				
				cells[a][b].weights[c] = Math.random();
				cells[a][b].lastWeights[c] = 0;
				
			}
			
		}
		
	}
	
}

function activate (x) {
	
	return 1 / (1 + Math.exp(-x));
	// return 0.1 * Math.sqrt(Math.abs(x)) * (x / Math.abs(x));
	// return 0.5 * x + Math.sin(x);
	
	// var y = Math.exp(2 * x);
	// 
	// return (y - 1) / (y + 1);
	
}

function feedForward (input) {
	
	if (cells[0].length - 1 != input.length) console.log("incorrect ff data");
	
	for (var a = 0; a < input.length; a++) {
		
		cells[0][a].value = input[a];
		
	}
	
	var sum = 0;
	
	for (var a = 1; a < cells.length; a++) {
		
		for (var b = 0; b < cells[a].length; b++) {
			
			sum = 0;
			
			for (var c = 0; c < cells[a - 1].length; c++) {
				
				sum += cells[a - 1][c].value * cells[a][b].weights[c];
				
			}
			
			// cells[a][b].value = activate(sum);
			cells[a][b].value = (a == cells.length - 1) ? sum : activate(sum);
			
		}
		
	}
	
}

function backpropagate (targets) {
	
	var cell;
	
	if (targets.length != cells[cells.length - 1].length) console.log("incorrect bp data");
	
	for (var a = 0; a < cells[cells.length - 1].length; a++) {
		
		cell = cells[cells.length - 1][a];
		
		cell.error = targets[a] - cell.value;
		// cell.error = cell.value * (1 - cell.value) * (targets[a] - cell.value);
		
	}
	
	var sum = 0;
	
	for (var a = cells.length - 2; a > 0; a--) {
		
		for (var b = 0; b < cells[a].length; b++) {
			
			sum = 0;
			cell = cells[a][b];
			
			for (var c = 0; c < cells[a + 1].length; c++) {
				
				sum += cells[a + 1][c].error * cells[a + 1][c].weights[b];
				
			}
			
			cell.error = cell.value * (1 - cell.value) * sum;
			
		}
		
	}
	
	for (var a = 1; a < cells.length; a++) {
		
		for (var b = 0; b < cells[a].length; b++) {
			
			cell = cells[a][b];
			
			for (var c = 0; c < cells[a - 1].length; c++) {
				
				cell.weights[c] += learningRate * cell.error * cells[a - 1][c].value;
				cell.lastWeights[c] = cell.weights[c];
				
			}
			
		}
		
	}
	
}

function ask (input) {
	
	feedForward(input);
	
	var ret = [];
	
	for (var a = 0; a < cells[cells.length - 1].length; a++) {
		
		ret.push(Math.round(10000 * cells[cells.length - 1][a].value) / 10000);
		
	}
	
	return ret;
	
}

function traceNetwork (callName) {
	
	var s = "";
	// var s = "####################\n" + callName + "\n####################\n";
	
	var precision = 6;
	
	for (var a = 0; a < cells.length; a++) {
		
		for (var b = 0; b < cells[a].length; b++) {
			
			s += "--- " + a + " X " + b;
			
			if (a == cells.length - 1) s += " output";
			else if (cells[a][b].bias) s += " bias";
			if (a == 0) s += " input";
			
			s += "\nv " + ("" + cells[a][b].value).slice(0, precision);
			
			if (cells[a][b].weights) {
				
				s += "\nw";
				
				for (var c = 0; c < cells[a][b].weights.length; c++) {
					
					s += " " + ("" + cells[a][b].weights[c]).slice(0, precision);
					
				}
				
				s += "\nd";
				
				for (var c = 0; c < cells[a][b].lastWeights.length; c++) {
					
					s += " " + ("" + cells[a][b].lastWeights[c]).slice(0, precision);
					
				}
				
				s += "\ne " + ("" + cells[a][b].error).slice(0, precision);
				
			}
			
			s += "\n";
			
		}
		
	}
	
	Art.doWrite(0, s);
	
}

function measureShannon () {
	
	var possibleStates = 16;
	var counts = [];
	
	for (var a = 0; a < possibleStates; a++) {
		
		counts.push(0);
		
	}
	
	for (var a = 0; a < sampleData.length; a++) {
		
		counts[sampleData[a]]++;
		
	}
	
	var shannonEntropy = 0;
	
	for (var a = 0; a < possibleStates; a++) {
		
		if (counts[a] == 0) continue;
		
		shannonEntropy -= (counts[a] / sampleData.length) * (Math.log(counts[a] / sampleData.length) / Math.log(possibleStates));
		
	}
	
	return shannonEntropy;
	
}

function storeSample () {
	
	// store the sample as binary
	sampleData.push(Math.round(ask([0, 0])[0]) * 1 +
					Math.round(ask([0, 1])[0]) * 2 +
					Math.round(ask([1, 0])[0]) * 4 +
					Math.round(ask([1, 1])[0]) * 8);
	
}

Stecy.setup = function () {
	
	Art.title = "PrimitiveNN";
	
	Input.mouseDefaultEnabled = true;
	
};

Art.ready = function () {
	
	momentum = 0.6;
	learningRate = 0.1;
	iterations = 100000;
	
	buildBrain([2, 10, 10, 1]);
	
	for (var a = 0; a < iterations + 1; a++) {
		
		var v = [Math.round(Math.random()),
				Math.round(Math.random())];
		
		feedForward(v);
		
		backpropagate([(v[0] && v[1]) || (!v[0] && !v[1])]);
		
		storeSample();
		
		if (a % 100 == 0 && a > 0) {
			
			// Art.doWrite(0, measureShannon().toFixed(4) + (a == iterations ? "" : ", "));
			
			sampleData = [];
			
		}
		
	}
	
	traceNetwork();
	
	// Art.doWrite(0, "\n\n");
	// Art.doWrite(0, "0, 0 > " + ask([0, 0]).join(", ") + "\n");
	// Art.doWrite(0, "0, 1 > " + ask([0, 1]).join(", ") + "\n");
	// Art.doWrite(0, "1, 0 > " + ask([1, 0]).join(", ") + "\n");
	// Art.doWrite(0, "1, 1 > " + ask([1, 1]).join(", ") + "\n");
	
};