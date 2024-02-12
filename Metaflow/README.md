# MetaFlow

- Metaflow is a python framework originally developed by Netflix to help machine learning practitioners to boost their productivity.
  - It is aimed to simplify the process of building and managing data science projects by providing tools and abstractions for key tasks like data processing, model training, artefact management (versioning and storage), and deployment

- Some of its key features
  - Parameters handling (Add arguments to flows): No need to use another library (e.g., argparse) to handle script parameters
    - [More about parameters usages](https://docs.metaflow.org/metaflow/basics#how-to-define-parameters-for-flows)
  - Versioning runs and artefacts (data, models, ...)
  - Store artifacts and outputs 
  - Parallelize code 
  - Connect flows or workflow orchestration: it enables to run an ML project as series of steps

## Command line interface

- To see main commands (`configure`, `status`, `tutorials`, ...)
  - Run `metaflow --help` command from the console/Terminal
- `tutorials` command
  - `metaflow tutorials --help` for more information
  - Copy tutorials in current directory: `metaflow tutorials pull`
  - List available tutorials: `metaflow tutorials list`
  - Get info on a particular tutorial: `metaflow tutorials info 00-helloworld`
- `show` command: show a brief summary of a flow (description, steps)
- `run` command
  - Run a flow from the command line: `python <MYSCRIPT.py> run`
- `resume` command
  - Debug failed flows: `python <MYSCRIPT.py> resume`
  - More on `resume` command: https://docs.metaflow.org/metaflow/debugging#how-to-use-the-resume-command


## Concepts

- Flow: A directed graph of operation called `step`
  - Every Flow needs a step called `start` and a step called `end` 
  - Supported transitions (to the next step)
    - `Linear`:  Moves from one step to another
    - `Branch`: Express parallel steps
      - `Static branch`: When we know steps (or branches) at development time.
      - `Foreach branch`: Useful when we do not know the branches at the development time or we when want to branch based on data dynamically.
- Step: An operation that we define (node of the graph). An individual units of execution.
- Run: An execution of a Flow
  - A `run` is successful if final `end` step finishes successfully
- Artefact
  - Go to (for more information) [https://docs.metaflow.org/metaflow/basics#artifacts](https://docs.metaflow.org/metaflow/basics#artifacts)

### Examples
- Minimum Flow
  - All `Flows` should inherit from the `FlowSpec`
  - Every `step` is decorated by `@step`
  
````python
from metaflow import FlowSpec, step

class MyFlow(FlowSpec):
    
    @step
    def start(self):
        self.next(self.end)
    
    @step
    def end(self):
        print("Flow is done!")

if __name__ == "__main__":
    MyFlow()
````

- `python my_small_flow.py run`
  - supposing that `my_small_flow.py` is the file name containing the code above.


  
## Examples 
When we run a Flow, Metaflow automatically create `.metaflow` directory under current working directory. This directory contains runs, Flows and artefacts.

- Train ElasticNet classifier to predict diabetics.
  - Run `python lr_flow.py run --help` command for details and brief summary
  - Run `python lr_flow.py show` command to see the Flow  (graph)
  - Run `python lr_flow.py run` to run the example with basic and default parameters
  - Run `python lr_flow.py run --alpha 0.02 --max-iter 300` to change default parameters values (i.e., for `alpha` and `max-iter`)



## Visualisation


### Cards
- Help to
  - attach customized visualizations to Flows
  - visualize and monitor model performances, 
  - compare performances across multiple runs
  - create report for data quality
  - monitor tasks progress
  - ...
- Can contain images, text, and tables that help observe flows.
- To create card:
  - from a given `step`: add `@card` decorator to the `step` from which we need to create report. This creates and store cards independently of Flows and Tasks.
    - This will not change the workflow behavior
  - attach `cards` to runs (i.e., to every `step`): Run your script with `--with card` option
    - `python lr_flow.py run --with card`
- Cards are stored (in Metaflow datastore, i.e., `.metaflow` directory) and versioned automatically
- Visualize cards
  - Example to visualize cards attached to `start` step: `python lr_flow.py card view start`
    - Run `python lr_flow.py card view --help` for more details and options
  - Local card viewer
    - Run `python lr_flow.py card server` command 
      - or Run it in background with this command: `nohup python lr_flow.py card server > nohup.out &` 
    - Then go to http://localhost:8324/ url to visualize runs, cards, artefacts, ...

  
## API reference
- Details about classes, decorators, ..., almost what you need know about `metaflow` library.
  - [https://docs.metaflow.org/api](https://docs.metaflow.org/api)


## Resources
- Official documentation: https://docs.metaflow.org/
- Install Metaflow: https://docs.metaflow.org/getting-started/install
- https://docs.metaflow.org/introduction/metaflow-resources
- For Data scientists: https://outerbounds.com/engineering/welcome/
- For ML/Data Engineers: https://outerbounds.com/engineering/welcome/