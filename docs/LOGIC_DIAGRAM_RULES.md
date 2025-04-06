# Logic Diagram Rules for Code Documentation

## Overview

This document defines the standards for creating and maintaining logic diagrams for code files. Every script in Python, JavaScript, TypeScript, or Rust must have an accompanying logic diagram that visualizes its high-level process flow.

## File Structure

### Base Directory
All logic diagrams must be stored in the `docs/mermaid` directory, maintaining a parallel structure to the source code:
```
project_root/
├── src/
│   ├── process_data.py
│   └── utils/
│       └── helper.ts
└── docs/
    └── mermaid/
        ├── process_data_logic.html
        └── utils/
            └── helper_logic.html
```

### Naming Convention
- For any source file `filename.{py|js|ts|rs}`, create a corresponding `filename_logic.html` in the parallel `docs/mermaid` directory structure
- Logic diagram files MUST use the `.html` extension (NOT `.md`) to ensure proper rendering
- Example:
  - Source file: `src/utils/data_processor.py`
  - Logic diagram: `docs/mermaid/utils/data_processor_logic.html`

### Directory Mirroring
- The `docs/mermaid` directory should mirror the source code directory structure
- Create subdirectories in `docs/mermaid` that match source code organization
- For modules with multiple files, create a `module_name_logic.html` in the corresponding mermaid directory

## Diagram Structure

### Required Sections

1. **Header**
   ```html
   <div class="row mb-4">
       <div class="col">
           <h1 class="display-4 text-center">[Script Name]</h1>
           <p class="lead text-center text-muted">[Brief Description]</p>
       </div>
   </div>
   ```

2. **Info Cards**
   - Key Features card
   - Process Statistics card
   - Must use Bootstrap card components

3. **Mermaid Diagram**
   - Must use the standard template structure
   - Include proper styling classes
   - Follow directional flow (top to bottom)

4. **Legend**
   - Must define all node types used
   - Use consistent colors across projects

5. **Timing Information**
   - Include process flow timing
   - Use Bootstrap table format

### Mermaid Diagram Standards

The following rules MUST be followed for all Mermaid.js diagrams:

1. **Basic Structure**
   ```html
   <div class="mermaid">
       graph TB
           %% Node Definitions
           Start([Start]) --> Process[Step]
           Process --> End([End])
           
           %% Style Definitions - NO semicolons
           classDef terminal fill:#bfb,stroke:#333,stroke-width:2px
           classDef process fill:#bbf,stroke:#333,stroke-width:2px
           
           %% Class Assignments - NO semicolons
           class Start,End terminal
           class Process process
   </div>
   ```

2. **Syntax Requirements**
   - NEVER use semicolons in `classDef` or `class` statements
   - Define all styles before class assignments
   - Group related nodes in class assignments using commas
   - Use proper node shapes:
     - Start/End: `([Text])`
     - Process: `[Text]`
     - Decision: `{Text}`

3. **Standard Colors**
   ```css
   terminal: fill:#bfb,stroke:#333,stroke-width:2px
   process: fill:#bbf,stroke:#333,stroke-width:2px
   decision: fill:#ffb,stroke:#333,stroke-width:2px
   default: fill:#f9f,stroke:#333,stroke-width:2px
   ```

4. **Required Comments**
   - Node Definitions: `%% Node Definitions`
   - Style Definitions: `%% Style Definitions`
   - Class Assignments: `%% Class Assignments`

5. **Common Errors to Avoid**
   ```mermaid
   %% ❌ WRONG:
   classDef process fill:#bbf;
   class Node1; Node2 process;
   
   %% ✅ CORRECT:
   classDef process fill:#bbf
   class Node1,Node2 process
   ```

## Language-Specific Guidelines

### Python
- Include async operations in separate swim lanes
- Show generator functions with dashed borders
- Highlight context managers

### JavaScript/TypeScript
- Show Promise chains clearly
- Indicate async/await flows
- Mark event handlers distinctly

### Rust
- Show ownership transfers
- Indicate lifetime boundaries
- Mark unsafe blocks distinctly

## Updating Guidelines

### When to Update
1. Any functional change to the source code
2. API signature changes
3. Flow control modifications
4. Error handling changes
5. New feature additions

### Update Process
1. Identify code changes
2. Update affected diagram sections
3. Validate diagram accuracy
4. Update timing information
5. Update process statistics

### Automation

#### Git Hooks
```bash
#!/bin/bash
# pre-commit hook
for file in $(git diff --cached --name-only); do
    if [[ $file =~ \.(py|js|ts|rs)$ ]]; then
        logic_file="${file%.*}_logic.html"
        if [[ -f $logic_file ]]; then
            echo "Please update $logic_file to reflect changes in $file"
        fi
    fi
done
```

#### IDE Integration
- VSCode extension recommendations
- Auto-detection of changes
- Diagram preview support

## Template Usage

### Basic Template
```html
<!DOCTYPE html>
<html>
<head>
    <title>[Script Name] - Logic Flow</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
    <script src="https://cdn.jsdelivr.net/npm/mermaid/dist/mermaid.min.js"></script>
    <!-- [Standard Styles] -->
</head>
<body>
    <!-- [Standard Sections] -->
</body>
</html>
```

### Diagram Sections
1. Input/Setup
2. Main Processing
3. Error Handling
4. Output/Cleanup
5. Validation Steps

## Best Practices

1. **Clarity**
   - Keep diagrams focused on high-level flow
   - Avoid implementation details
   - Use clear, consistent naming

2. **Maintainability**
   - Update diagrams with code changes
   - Keep styling consistent
   - Use standard templates

3. **Completeness**
   - Include all major process steps
   - Show error handling
   - Document timing expectations

4. **Validation**
   - Verify diagram accuracy
   - Check for broken flows
   - Ensure all nodes are connected

## Review Process

1. **Code Review Requirements**
   - Logic diagram updates must be included in PRs
   - Diagrams must be validated
   - Timing information must be accurate

2. **Quality Checks**
   - Diagram follows template
   - All sections present
   - Styling is consistent
   - Information is accurate

## Tools and Resources

1. **Required Tools**
   - Mermaid.js
   - Bootstrap 5.3+
   - Git hooks

2. **Recommended Extensions**
   - Mermaid Preview
   - Markdown All in One
   - Bootstrap Snippets

## Troubleshooting

1. **Common Issues**
   - Diagram not rendering
   - Inconsistent styling
   - Missing sections

2. **Solutions**
   - Validate Mermaid syntax
   - Check template usage
   - Verify file structure

## Standard Template Structure

### Complete HTML Template
```html
<!DOCTYPE html>
<html>
<head>
    <title>[Script Name] - Logic Flow</title>
    <!-- Required CSS -->
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
    <script src="https://cdn.jsdelivr.net/npm/mermaid/dist/mermaid.min.js"></script>
    
    <!-- Standard Styles - REQUIRED -->
    <style>
        body {
            background-color: #f8f9fa;
            padding-top: 2rem;
        }
        .diagram-container {
            background-color: white;
            border-radius: 1rem;
            box-shadow: 0 0.125rem 0.25rem rgba(0,0,0,0.075);
            margin-bottom: 2rem;
        }
        .info-card {
            margin-bottom: 1.5rem;
        }
        .legend-item {
            display: inline-block;
            padding: 0.25rem 0.75rem;
            margin: 0.25rem;
            border-radius: 0.25rem;
            font-size: 0.875rem;
        }
    </style>
</head>
<body>
    <div class="container">
        <!-- 1. Header Section -->
        <div class="row mb-4">
            <div class="col">
                <h1 class="display-4 text-center">[Script Name]</h1>
                <p class="lead text-center text-muted">[Brief Description]</p>
            </div>
        </div>

        <!-- 2. Info Cards Section -->
        <div class="row mb-4">
            <!-- Key Features Card -->
            <div class="col-md-6">
                <div class="card info-card">
                    <div class="card-header bg-primary text-white">
                        <h5 class="card-title mb-0">Key Features</h5>
                    </div>
                    <div class="card-body">
                        <ul class="list-group list-group-flush">
                            <li class="list-group-item">✨ [Feature 1]</li>
                            <li class="list-group-item">🔗 [Feature 2]</li>
                            <li class="list-group-item">🔄 [Feature 3]</li>
                            <li class="list-group-item">✅ [Feature 4]</li>
                        </ul>
                    </div>
                </div>
            </div>
            <!-- Process Stats Card -->
            <div class="col-md-6">
                <div class="card info-card">
                    <div class="card-header bg-success text-white">
                        <h5 class="card-title mb-0">Process Statistics</h5>
                    </div>
                    <div class="card-body">
                        <ul class="list-group list-group-flush">
                            <li class="list-group-item">⚡ Processing Time: [Time]</li>
                            <li class="list-group-item">📊 Success Rate: [Rate]</li>
                            <li class="list-group-item">🔄 Retry Count: [Count]</li>
                            <li class="list-group-item">📝 [Other Stat]</li>
                        </ul>
                    </div>
                </div>
            </div>
        </div>

        <!-- 3. Mermaid Diagram Section -->
        <div class="diagram-container p-4">
            <div class="mermaid">
                graph TB
                    %% Your diagram code here
                    %% Use standard node types and styling
            </div>

            <!-- 4. Legend Section -->
            <div class="mt-4">
                <h5 class="text-muted mb-3">Legend</h5>
                <div class="legend-item" style="background-color: #bbf">Data Nodes</div>
                <div class="legend-item" style="background-color: #f9f">Process Nodes</div>
                <div class="legend-item" style="background-color: #bfb">Operation Nodes</div>
                <div class="legend-item" style="background-color: #ffb">Validation Steps</div>
            </div>
        </div>

        <!-- 5. Timing Information Section -->
        <div class="card mb-4">
            <div class="card-header bg-info text-white">
                <h5 class="card-title mb-0">Process Timing</h5>
            </div>
            <div class="card-body">
                <div class="table-responsive">
                    <table class="table table-sm">
                        <thead>
                            <tr>
                                <th>Step</th>
                                <th>Time</th>
                                <th>Description</th>
                            </tr>
                        </thead>
                        <tbody>
                            <tr>
                                <td>t=0</td>
                                <td>[Step Name]</td>
                                <td>[Step Description]</td>
                            </tr>
                            <!-- Add more timing rows as needed -->
                        </tbody>
                    </table>
                </div>
            </div>
        </div>
    </div>

    <!-- Required JavaScript -->
    <script src="https://cdn.jsdelivr.net/npm/@popperjs/core@2.11.6/dist/umd/popper.min.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/js/bootstrap.min.js"></script>
    
    <!-- Mermaid Initialization -->
    <script>
        mermaid.initialize({
            startOnLoad: true,
            theme: 'default',
            flowchart: {
                useMaxWidth: true,
                htmlLabels: true,
                curve: 'basis'
            }
        });
    </script>
</body>
</html>
```

### Required Color Scheme
```css
/* Standard Colors */
:root {
    --node-data: #bbf;
    --node-process: #f9f;
    --node-operation: #bfb;
    --node-validation: #ffb;
    --header-bg: #f8f9fa;
    --card-shadow: rgba(0,0,0,0.075);
}
```

### Component Guidelines

1. **Info Cards**
   - Must use Bootstrap card components
   - Left card: Key Features (bg-primary)
   - Right card: Process Stats (bg-success)
   - Use emojis for visual enhancement

2. **Diagram Container**
   - White background
   - Rounded corners (1rem)
   - Subtle shadow
   - Proper padding (p-4)

3. **Legend Items**
   - Consistent colors with diagram
   - Rounded corners
   - Inline-block display
   - Standard padding and margins

4. **Timing Table**
   - Bootstrap responsive table
   - Three columns: Step, Time, Description
   - Light styling (table-sm)
   - Info header (bg-info)

[... rest of the existing content ...] 