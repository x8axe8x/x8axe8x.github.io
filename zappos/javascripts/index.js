var armRewards = [-3, -2, -1, 0, 1, 2, 3];  //initialize with arm reward means
var alpha = .1;             //alpha: learning step size
var numTasks = 500;        //number of 7-arm bandits
var numActions = 500;      //number of arm pulls

var ept = function(e){      //halves epsilon after every 1000 steps
  return function(t){e/Math.pow(2, Math.floor(t/1000))};
};


//initialize reward array of Gaussians with given mean and variance of 1
for (mean in armRewards){
  armRewards[mean] = d3.random.normal(armRewards[mean], 1);    
};

var optAction=[]; //holds count of optimal action selected per time step

for (var k=0; k<numActions; ++k){
  optAction.push(0);
};

var action;
var reward;
var actionVals = [0,0,0,0,0,0,0];   //Q_t(a)

//tasks are done in batches of numTasks/numBatches in order to prevent browser from detecting false infinite loops...
var eGreedBatch = function(ep, numSteps, numBatches){
  for(var j=0; j<numTasks/numBatches; ++j){    
    actionVals = [1,0,0,0,0,0,0];   //initialize all estimate action values to zero for each task

    for (var i=0; i<numSteps; ++i){
      if(Math.random()<= (_.isFunction(ep)?ep(i):ep)){      //explore
        action = Math.floor(Math.random()*7); //choose a random arm to pull
      }
      else {                      //exploit highest value action
        var best = d3.max(actionVals);
        var bests = actionVals.filter(function(element){
          return element == best;
        });
        best = d3.shuffle(bests)[0];    //randomly select arm if multiple maxima
        action = actionVals.indexOf(best);
      };

      reward = armRewards[action]();  //get reward for arm selected

      actionVals[action] = actionVals[action] + alpha*(reward - actionVals[action]);    //update reward value

      if (action == 6) {optAction[i]++};  //increment optimal action selected (arm 7) for this time step
    }
  }
}

//divides the total tasks into 8 batches of numTasks/8
var eGreed = function(ep, numSteps){
  eGreedBatch(ep, numSteps, 8);
  eGreedBatch(ep, numSteps, 8);
  eGreedBatch(ep, numSteps, 8);
  eGreedBatch(ep, numSteps, 8);
  eGreedBatch(ep, numSteps, 8);
  eGreedBatch(ep, numSteps, 8);
  eGreedBatch(ep, numSteps, 8);
  eGreedBatch(ep, numSteps, 8);
};

var pi = function(tau){   //returns softmax action selection probability array 
  var sum = d3.sum(actionVals, function(val){
    return Math.exp(val/tau)
  });
    
  return _.map(actionVals, function(val){
    return Math.exp(val/tau)/sum;
  });
};

var getA = function(pi_t){   //returns selected action by generating a random number in the prob distribution
  var sel = Math.random();  //generate a random number to select from pi_t distr
  var sum = 0;
  
  for (a in pi_t){
    if (sel <= (sum+= pi_t[a])) 
      return a; //if selection falls in this area, select it
  }
};

var totalRew;
var pi_t;

var softmaxBatch = function(tau, numActions, numBatches){
  for(var j=0; j<numTasks/numBatches; ++j){    //tasks are done in batches of numTasks/div in order to prevent browser from detecting false infinite loops...
    actionVals = [0,0,0,0,0,0,0];   //initialize all estimate action values to zero for each task
    totalRew = 0;
    for (var i=0; i<numActions; ++i){
      action = getA(pi_t=pi(tau));
      
      totalRew+= reward = armRewards[action]();  //get reward for arm selected

      actionVals[action] = actionVals[action] + alpha*(reward - totalRew/(i+1))*(1-pi_t[action]);    //update reward value

      for (act in actionVals){
        if(act!=action)
          actionVals[act] = actionVals[act] - alpha*(reward - totalRew/(i+1))*pi_t[act];
      }
      if (action == 6) {optAction[i]++};  //increment optimal action selected (arm 7) for this time step
    }
  }
}
//batches for softmax are in tenths (more data intensive)
var softmax = function(tau, numSteps){
  softmaxBatch(tau, numSteps, 10);
  softmaxBatch(tau, numSteps, 10);
  softmaxBatch(tau, numSteps, 10);
  softmaxBatch(tau, numSteps, 10);
  softmaxBatch(tau, numSteps, 10);
  softmaxBatch(tau, numSteps, 10);
  softmaxBatch(tau, numSteps, 10);
  softmaxBatch(tau, numSteps, 10);
  softmaxBatch(tau, numSteps, 10);
  softmaxBatch(tau, numSteps, 10);
};
//just sets up a plot for the resulting data (selects an <SVG> html element with id="plot")
var genPlot = function(){
  var lineData=[];
  for (action in optAction){
    lineData.push({
      x: action,
      y: optAction[action]/numTasks*100   //percent optimal action selection
    });
  }

  var vis = d3.select('#plot'),
      WIDTH = 1000,
      HEIGHT = 500,
      MARGINS = {
        top: 30,
        right: 20,
        bottom: 50,
        left: 50
      },
      xRange = d3.scale.linear().range([MARGINS.left, WIDTH - MARGINS.right]).domain([0,numActions]),
      yRange = d3.scale.linear().range([HEIGHT - MARGINS.bottom, MARGINS.top]).domain([0,100]),
      xAxis = d3.svg.axis()
        .scale(xRange)
        .tickSize(1)
        .tickSubdivide(true),
      yAxis = d3.svg.axis()
        .scale(yRange)
        .tickSize(1)
        .orient('left')
        .tickSubdivide(true);

  vis.append('svg:g')
    .attr('class', 'x axis')
    .attr('transform', 'translate(0,' + (HEIGHT - MARGINS.bottom) + ')')
    .call(xAxis);

  vis.append('svg:g')
    .attr('class', 'y axis')
    .attr('transform', 'translate(' + (MARGINS.left) + ',0)')
    .call(yAxis);

  vis.append('svg:text').attr("x",WIDTH/2).attr("y", HEIGHT-10).attr("text-anchor", "middle").text("Arm Pulls");

  vis.append('svg:text').attr("text-anchor", "middle").attr("transform", "rotate(-90)").attr("y", 20).attr("x",-HEIGHT/2).text("% Optimal Arm Selected");

  var lineFunc = d3.svg.line()
    .x(function(d) {
      return xRange(d.x);
    })
    .y(function(d) {
      return yRange(d.y);
    })
    .interpolate('linear');

  vis.append('svg:path')
    .attr('d', lineFunc(lineData))
    .attr('stroke', 'purple')
    .attr('stroke-width', 1)
    .attr('fill', 'none');
};
//setInterval(genPlot, 10000);

//set the global numActions at top (this is so plot generates correctly)
////eGreed(.01, numActions);    //E-Greedy method with constant Epsilon
//eGreed(ept(.9), numActions);      //E-Greedy method with Epsilon halving every 1000 actions
softmaxBatch(1, numActions, 1);         //softmax/Gibbs learning algorithm
genPlot();