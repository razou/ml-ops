# What is Prefect

- **Prefect** is an orchestration and observability platform designed for building, observing, and managing workflows. It simplifies the process of transforming Python code into interactive workflow applications. With Prefect, you can expose your workflows through an API, allowing teams to programmatically access your pipelines and business logic.
- Official documentation: [https://docs.prefect.io/latest/](https://docs.prefect.io/latest/)


# Some important concepts to know
- [Task](https://docs.prefect.io/latest/concepts/tasks/): Unit of work in prefect (e.g., a function)
  - [https://docs.prefect.io/latest/tutorial/tasks/](https://docs.prefect.io/latest/tutorial/tasks/)
- [Flow](https://docs.prefect.io/latest/concepts/flows/): A container for workflow logic, defined as a Python function.
  - [https://docs.prefect.io/latest/tutorial/flows/](https://docs.prefect.io/latest/tutorial/flows/)
- [Deployment](https://docs.prefect.io/latest/concepts/deployments/): 
  - As per the official documentation: *Deployments are server-side representations of flows. They store the crucial metadata needed for remote orchestration including when, where, and how a workflow should run.*
  - [https://docs.prefect.io/latest/tutorial/deployments/](https://docs.prefect.io/latest/tutorial/deployments/)
- Flow run: A single execution of a flow.
- [Artifact](https://docs.prefect.io/latest/concepts/artifacts/): Persisted results (e.g., tables, links, Markdown)
- [Schedules](https://docs.prefect.io/latest/concepts/schedules/):They allow to automatically create new flow runs for deployments.


# What it allows to do

1. **Workflow Creation and Orchestration**:
   - Design, build, and manage complex workflows using Python code. You can transform any Python function into a unit of work (i.e., `task`) that can be orchestrated.
   - Define dependencies between tasks, handle retries, and schedule execution (or deployment).

2. **Scheduling and Automation**:
   - Prefect provides scheduling capabilities, allowing you to automate the execution of tasks and workflows.
   - It supports different types of schedules:
     - **[Cron](https://docs.prefect.io/latest/concepts/schedules/#cron)**
       -  [https://en.wikipedia.org/wiki/Cron](https://en.wikipedia.org/wiki/Cron)
     - **[Interval](https://docs.prefect.io/latest/concepts/schedules/#interval)**
     - **[RRule](https://docs.prefect.io/latest/concepts/schedules/#rrule)**

3. **Monitoring and Observability**:
   - Prefect offers tools for tracking and visualizing workflow execution.
   - Monitor task status, performance, and resource utilization.
   - React to issues and failures in real time.

4. **Error Handling and Retries**:
   - Prefect handles error scenarios gracefully. You can configure retries, set error thresholds, and define custom error handling logic.

5. **Scalability and Concurrency Control**:
   - Prefect scales horizontally, allowing you to handle large-scale workflows.
   - Control concurrency and parallelism to optimize resource utilization.

6. **API Exposure**:
   - Expose your workflows through an API, enabling programmatic access to your pipelines and business logic.

    
# How to use it

1. **Install Prefect**:
   - Install sdk client using the following command:
     ```shell
     pip install -U prefect
     ```

2. **Setup and/or connect to a Prefect server instance**:
   - The server makes it easy to monitor and execute flows
   - It allows keep track and to persist record of your runs, current state
   - It also allows asynchronous scheduling and notifications
   - It is backed by a database and a UI
     - The database persists data to track the state of flow runs, run history, artifacts, logs , deployments, variables,...
     - Supported databases (Prefect version `2.18.1`): 
       - **SQLite** (default): It is configured during the prefect install and is located at `~/.prefect/prefect.db` by default.
         - Type this command if you want to reset it (i.e., clear all data):`prefect server database reset -y`
       - **PostgreSQL**

   - There are different possibilities to set Prefect server instance including:
     - Local mode (for testing and debuging purposes)
     - Using Prefect Cloud (Cloud service hosted by Prefect)
       - [https://docs.prefect.io/latest/cloud/](https://docs.prefect.io/latest/cloud/)
       - Account types and prices: [https://www.prefect.io/pricing](https://www.prefect.io/pricing)
     - Setting Prefect server instance on Kubernetes
       - [https://github.com/PrefectHQ/prefect-helm/tree/main/charts/prefect-server](https://github.com/PrefectHQ/prefect-helm/tree/main/charts/prefect-server)

   - Example: **In local mode**
     - Start the server by the following command: `prefect server start` 
     - The go to [http://127.0.0.1:4200/](http://127.0.0.1:4200/) to get access to the server UI

3. **Create Your Workflow**:
   - Example:
     ```python
     from prefect import task, flow

     @task
     def show_lib_version():
         import prefect
         print(f"Prefect version: {prefect.__version__}")

     @flow
     def show_version_flow():
         show_lib_version() 
     ```

4. **Choose a Remote Infrastructure Location**:
   - Decide where you want to run your workflows. You can choose between self-hosting an open-source Prefect server or using Prefect Cloud for additional features.

5. **Set up schedules**:
   - Using decorators or configuration files.

   
# Example (Local usage)

## What to do
- Train and compare different classifier models:
  - Train models in parallel using `ConcurrentTaskRunner`
- Log Performances' table as an [Artefact](https://docs.prefect.io/latest/concepts/artifacts/)
- Log best Model (serialized into ... format) as a [Result](https://docs.prefect.io/latest/concepts/results/):
  -  `~/.prefect/storage` is the local storage path by default
    - Run `ls ~/.prefect/storage` command to list its content
  

## How to do it

1. Start prefect server: `prefect server start`
   - This may trigger some errors like:
    1. `sqlite3.OperationalError: database is locked`
       - [https://linen.prefect.io/t/12329117/hi-everyone-i-am-new-to-prefect-and-i-tried-following-the-ge](https://linen.prefect.io/t/12329117/hi-everyone-i-am-new-to-prefect-and-i-tried-following-the-ge)
       - [https://github.com/PrefectHQ/prefect/issues/10188](https://github.com/PrefectHQ/prefect/issues/10188)
         - [https://github.com/PrefectHQ/prefect/issues/7277](https://github.com/PrefectHQ/prefect/issues/7277)
   2. `sqlalchemy.exc.OperationalError: (sqlite3.OperationalError) no such table: json_each`
       - [https://github.com/PrefectHQ/prefect/issues/5970](https://github.com/PrefectHQ/prefect/issues/5970)
       - [https://github.com/PrefectHQ/prefect/issues/7916](https://github.com/PrefectHQ/prefect/issues/7916)

2. UI available at: [http://127.0.0.1:4200/](http://127.0.0.1:4200/)

3. Create single Flow Run with `create_flow_run.py` script
   - Run `python create_flow_run.py` command
4. Create deployment with Flow's `.serve()` method:
   - Run `ptython crate_deployment.py` command
     - This will return:
    ```shell
        To trigger a run for this flow, use the following command:
    
            $ prefect deployment run 'train-flow/TrainClassifiers'
    ```

   - Then run `prefect deployment run 'train-flow/TrainClassifiers'` command in another terminal to trigger a run for this flow.
    
    









