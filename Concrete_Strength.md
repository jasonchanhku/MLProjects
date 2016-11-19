Modeling Concrete Strength with ANN
================
Jason Chan
November 19, 2016

-   Libraries Used
-   Objective
-   Step 1: Data Exploration
    -   Data Preview
    -   Data Structure
    -   Features
    -   Data Visualization
        -   Strength Histogram
        -   Scatterplot Matrices
-   Step 2: Data Preprocessing & Preparation
    -   Traditional Normalization
    -   Data Preparation
-   Step 3: Model Training
    -   Visualizing Neural Network
-   Step 4: Model Evaluation
    -   Model Accuracy
-   Step 5: Improving the Model
    -   Visualizing the Improved Neural Network
    -   Implementing the Improved Neural Network
    -   Evaluating New Model

For the fully functional html version, please visit <http://www.rpubs.com/jasonchanhku/concrete>

Libraries Used
==============

``` r
library(ggvis) #Data visulization
library(psych) #Scatterplot matrix
library(knitr) #html table
library(neuralnet) #artifical neural network 
```

<br />

Objective
=========

In the engineering, it is crucial to have accurate estimates of concrete strength to develop safety guidelines in construction. Concrete performance varies greatly due to a wide variety of ingredients that interact in complex ways. As a result, it is difficult to accurately predict the strength of the final product.

This project aims to develop a reliable model using **Artificial Neural Networks (ANN)** to predict concrete strength given a list of composition inputs.

<br />

Step 1: Data Exploration
========================

The dataset on the compressive strength of concrete is obtained from the UCI Machine Learning Data Repository <http://archive.ics.uci.edu/ml>. The concrete dataset contains 1,030 examples of concrete with eight features describing the components used in the mixture.

Data Preview
------------

``` r
concrete <- read.csv(file = "Machine-Learning-with-R-datasets-master/concrete.csv")

knitr::kable(head(concrete), caption = "Partial Table Preview")
```

|  cement|   slag|  ash|  water|  superplastic|  coarseagg|  fineagg|  age|  strength|
|-------:|------:|----:|------:|-------------:|----------:|--------:|----:|---------:|
|   540.0|    0.0|    0|    162|           2.5|     1040.0|    676.0|   28|     79.99|
|   540.0|    0.0|    0|    162|           2.5|     1055.0|    676.0|   28|     61.89|
|   332.5|  142.5|    0|    228|           0.0|      932.0|    594.0|  270|     40.27|
|   332.5|  142.5|    0|    228|           0.0|      932.0|    594.0|  365|     41.05|
|   198.6|  132.4|    0|    192|           0.0|      978.4|    825.5|  360|     44.30|
|   266.0|  114.0|    0|    228|           0.0|      932.0|    670.0|   90|     47.03|

Data Structure
--------------

``` r
str(concrete)
```

    ## 'data.frame':    1030 obs. of  9 variables:
    ##  $ cement      : num  540 540 332 332 199 ...
    ##  $ slag        : num  0 0 142 142 132 ...
    ##  $ ash         : num  0 0 0 0 0 0 0 0 0 0 ...
    ##  $ water       : num  162 162 228 228 192 228 228 228 228 228 ...
    ##  $ superplastic: num  2.5 2.5 0 0 0 0 0 0 0 0 ...
    ##  $ coarseagg   : num  1040 1055 932 932 978 ...
    ##  $ fineagg     : num  676 676 594 594 826 ...
    ##  $ age         : int  28 28 270 365 360 90 365 28 28 28 ...
    ##  $ strength    : num  80 61.9 40.3 41 44.3 ...

Features
--------

**Target Variable:**

-   strength

**Features Used:**

-   cement
-   slag
-   ash
-   water
-   superplasticizer
-   coarse aggregate
-   fine aggregate

Data Visualization
------------------

Visualizing the data enables us to adjust the model if needed and to spot outliers at an early stage. It is also important to visualize the distribution of the target variable, strength.

### Strength Histogram

``` r
concrete %>% ggvis(x = ~strength, fill:= "#27bc9c") %>% layer_histograms() %>% layer_paths(y = ~strength, 35.82, stroke := "red")
```

<!--html_preserve-->

<nav class="ggvis-control"> <a class="ggvis-dropdown-toggle" title="Controls" onclick="return false;"></a>
<ul class="ggvis-dropdown">
<li>
Renderer: <a id="plot_id285353667_renderer_svg" class="ggvis-renderer-button" onclick="return false;" data-plot-id="plot_id285353667" data-renderer="svg">SVG</a> | <a id="plot_id285353667_renderer_canvas" class="ggvis-renderer-button" onclick="return false;" data-plot-id="plot_id285353667" data-renderer="canvas">Canvas</a>
</li>
<li>
<a id="plot_id285353667_download" class="ggvis-download" data-plot-id="plot_id285353667">Download</a>
</li>
</ul>
</nav>

<script type="text/javascript">
var plot_id285353667_spec = {
  "data": [
    {
      "name": ".0",
      "format": {
        "type": "csv",
        "parse": {
          "strength": "number"
        }
      },
      "values": "\"strength\"\n79.99\n61.89\n40.27\n41.05\n44.3\n47.03\n43.7\n36.45\n45.85\n39.29\n38.07\n28.02\n43.01\n42.33\n47.81\n52.91\n39.36\n56.14\n40.56\n42.62\n41.84\n28.24\n8.06\n44.21\n52.52\n53.3\n41.15\n52.12\n37.43\n38.6\n55.26\n52.91\n41.72\n42.13\n53.69\n38.41\n30.08\n37.72\n42.23\n36.25\n50.46\n43.7\n39\n53.1\n41.54\n35.08\n15.05\n40.76\n26.26\n32.82\n39.78\n46.93\n33.12\n49.19\n14.59\n14.64\n41.93\n9.13\n50.95\n33.02\n54.38\n51.73\n9.87\n50.66\n48.7\n55.06\n44.7\n30.28\n40.86\n71.99\n34.4\n28.8\n33.4\n36.3\n29\n37.8\n40.2\n33.4\n28.1\n41.3\n33.4\n25.2\n41.1\n35.3\n28.3\n28.6\n35.3\n24.4\n35.3\n39.3\n40.6\n35.3\n24.1\n46.2\n42.8\n49.2\n46.8\n45.7\n55.6\n54.9\n49.2\n34.9\n46.9\n49.2\n33.4\n54.1\n55.9\n49.8\n47.1\n55.9\n38\n55.9\n56.1\n59.09\n22.9\n35.1\n61.09\n59.8\n60.29\n61.8\n56.7\n68.3\n66.9\n60.29\n50.7\n56.4\n60.29\n55.5\n68.5\n71.3\n74.7\n52.2\n71.3\n67.7\n71.3\n66\n74.5\n71.3\n49.9\n63.4\n64.9\n64.3\n64.9\n60.2\n72.3\n69.3\n64.3\n55.2\n58.8\n64.3\n66.1\n73.7\n77.3\n80.2\n54.9\n77.3\n72.99\n77.3\n71.7\n79.4\n77.3\n59.89\n64.9\n66.6\n65.2\n66.7\n62.5\n74.19\n70.7\n65.2\n57.6\n59.2\n65.2\n68.1\n75.5\n79.3\n56.5\n79.3\n76.8\n79.3\n73.3\n82.6\n79.3\n67.8\n11.58\n24.45\n24.89\n29.45\n40.71\n10.38\n22.14\n22.84\n27.66\n34.56\n12.45\n24.99\n25.72\n33.96\n37.34\n15.04\n21.06\n26.4\n35.34\n40.57\n12.47\n20.92\n24.9\n34.2\n39.61\n10.03\n20.08\n24.48\n31.54\n35.34\n9.45\n22.72\n28.47\n38.56\n40.39\n10.76\n25.48\n21.54\n28.63\n33.54\n7.75\n17.82\n24.24\n32.85\n39.23\n18\n30.39\n45.71\n50.77\n53.9\n13.18\n17.84\n40.23\n47.13\n49.97\n13.36\n22.32\n24.54\n31.35\n40.86\n19.93\n25.69\n30.23\n39.59\n44.3\n13.82\n24.92\n29.22\n38.33\n42.35\n13.54\n26.31\n31.64\n42.55\n42.92\n13.33\n25.37\n37.4\n44.4\n47.74\n19.52\n31.35\n38.5\n45.08\n47.82\n15.44\n26.77\n33.73\n42.7\n45.84\n17.22\n29.93\n29.65\n36.97\n43.58\n13.12\n24.43\n32.66\n36.64\n44.21\n13.62\n21.6\n27.77\n35.57\n45.37\n7.32\n21.5\n31.27\n43.5\n48.67\n7.4\n23.51\n31.12\n39.15\n48.15\n22.5\n34.67\n34.74\n45.08\n48.97\n23.14\n41.89\n48.28\n51.04\n55.64\n22.95\n35.23\n39.94\n48.72\n52.04\n21.02\n33.36\n33.94\n44.14\n45.37\n15.36\n28.68\n30.85\n42.03\n51.06\n21.78\n42.29\n50.6\n55.83\n60.95\n23.52\n42.22\n52.5\n60.32\n66.42\n23.8\n38.77\n51.33\n56.85\n58.61\n21.91\n36.99\n47.4\n51.96\n56.74\n17.57\n33.73\n40.15\n46.64\n50.08\n17.37\n33.7\n45.94\n51.43\n59.3\n30.45\n47.71\n63.14\n66.82\n66.95\n27.42\n35.96\n55.51\n61.99\n63.53\n18.02\n38.6\n52.2\n53.96\n56.63\n15.34\n26.05\n30.22\n37.27\n46.23\n16.28\n25.62\n31.97\n36.3\n43.06\n67.57\n57.23\n81.75\n64.02\n78.8\n41.37\n60.28\n56.83\n51.02\n55.55\n44.13\n39.38\n55.65\n47.28\n44.33\n52.3\n49.25\n41.37\n29.16\n39.4\n39.3\n67.87\n58.52\n53.58\n59\n76.24\n69.84\n14.4\n19.42\n20.73\n14.94\n21.29\n23.08\n15.52\n15.82\n12.55\n8.49\n15.61\n12.18\n11.98\n16.88\n33.09\n34.24\n31.81\n29.75\n33.01\n32.9\n29.55\n19.42\n24.66\n29.59\n24.28\n20.73\n26.2\n46.39\n39.16\n41.2\n33.69\n38.2\n41.41\n37.81\n24.85\n27.22\n44.64\n37.27\n33.27\n36.56\n53.72\n48.59\n51.72\n35.85\n53.77\n53.46\n48.99\n31.72\n39.64\n51.26\n43.39\n39.27\n37.96\n55.02\n49.99\n53.66\n37.68\n56.06\n56.81\n50.94\n33.56\n41.16\n52.96\n44.28\n40.15\n57.03\n44.42\n51.02\n53.39\n35.36\n25.02\n23.35\n52.01\n38.02\n39.3\n61.07\n56.14\n55.25\n54.77\n50.24\n46.68\n46.68\n22.75\n25.51\n34.77\n36.84\n45.9\n41.67\n56.34\n47.97\n61.46\n44.03\n55.45\n55.55\n57.92\n25.61\n33.49\n59.59\n29.55\n37.92\n61.86\n62.05\n32.01\n72.1\n39\n65.7\n32.11\n40.29\n74.36\n21.97\n9.85\n15.07\n23.25\n43.73\n13.4\n24.13\n44.52\n62.94\n59.49\n25.12\n23.64\n35.75\n38.61\n68.75\n66.78\n23.85\n32.07\n11.65\n19.2\n48.85\n39.6\n43.94\n34.57\n54.32\n24.4\n15.62\n21.86\n10.22\n14.6\n18.75\n31.97\n23.4\n25.57\n41.68\n27.74\n8.2\n9.62\n25.42\n15.69\n27.94\n32.63\n17.24\n19.77\n39.44\n25.75\n33.08\n24.07\n21.82\n21.07\n14.84\n32.05\n11.96\n25.45\n22.49\n25.22\n39.7\n13.09\n38.7\n7.51\n17.58\n21.18\n18.2\n17.2\n22.63\n21.86\n12.37\n25.73\n37.81\n21.92\n33.04\n14.54\n26.91\n8\n31.9\n10.34\n19.77\n37.44\n11.48\n24.44\n17.6\n10.73\n31.38\n13.22\n20.97\n27.04\n32.04\n35.17\n36.45\n38.89\n6.47\n12.84\n18.42\n21.95\n24.1\n25.08\n21.26\n25.97\n11.36\n31.25\n32.33\n33.7\n9.31\n26.94\n27.63\n29.79\n34.49\n36.15\n12.54\n27.53\n32.92\n9.99\n7.84\n12.25\n11.17\n17.34\n17.54\n30.57\n14.2\n24.5\n15.58\n26.85\n26.06\n38.21\n43.7\n30.14\n12.73\n20.87\n20.28\n34.29\n19.54\n47.71\n43.38\n29.89\n6.9\n33.19\n4.9\n4.57\n25.46\n24.29\n33.95\n11.41\n20.59\n25.89\n29.23\n31.02\n10.39\n33.66\n27.87\n19.35\n11.39\n12.79\n39.32\n4.78\n16.11\n43.38\n20.42\n6.94\n15.03\n13.57\n32.53\n15.75\n7.68\n38.8\n33\n17.28\n24.28\n24.05\n36.59\n50.73\n13.66\n14.14\n47.78\n2.33\n16.89\n23.52\n6.81\n39.7\n17.96\n32.88\n22.35\n10.79\n7.72\n41.68\n9.56\n6.88\n50.53\n17.17\n30.44\n9.73\n3.32\n26.32\n43.25\n6.28\n32.1\n36.96\n54.6\n21.48\n9.69\n8.37\n39.66\n10.09\n4.83\n10.35\n43.57\n51.86\n11.85\n17.24\n27.83\n35.76\n38.7\n14.31\n17.44\n31.74\n37.91\n39.38\n15.87\n9.01\n33.61\n40.66\n40.86\n12.05\n17.54\n18.91\n25.18\n30.96\n43.89\n54.28\n36.94\n14.5\n22.44\n12.64\n26.06\n33.21\n36.94\n44.09\n52.61\n59.76\n67.31\n69.66\n71.62\n74.17\n18.13\n22.53\n27.34\n29.98\n31.35\n32.72\n6.27\n14.7\n23.22\n27.92\n31.35\n39\n41.24\n14.99\n13.52\n24\n37.42\n11.47\n22.44\n21.16\n31.84\n14.8\n25.18\n17.54\n14.2\n21.65\n29.39\n13.52\n16.26\n31.45\n37.23\n18.13\n32.72\n39.49\n41.05\n42.13\n18.13\n26.74\n61.92\n47.22\n51.04\n55.16\n41.64\n13.71\n19.69\n31.65\n19.11\n39.58\n48.79\n24\n37.42\n11.47\n19.69\n14.99\n27.92\n34.68\n37.33\n38.11\n33.8\n42.42\n48.4\n55.94\n58.78\n67.11\n20.77\n25.18\n29.59\n21.75\n39.09\n24.39\n50.51\n74.99\n37.17\n33.76\n16.5\n19.99\n36.35\n33.69\n15.42\n33.42\n39.05\n27.68\n26.86\n45.3\n30.12\n15.57\n44.61\n53.52\n57.21\n65.91\n52.82\n33.4\n18.03\n37.36\n32.84\n42.64\n40.06\n41.94\n61.23\n40.87\n33.3\n52.42\n15.09\n38.46\n37.26\n35.23\n42.13\n31.87\n41.54\n39.45\n37.91\n44.28\n31.18\n23.69\n32.76\n32.4\n28.63\n36.8\n18.28\n33.06\n31.42\n31.03\n44.39\n12.18\n25.56\n36.44\n32.96\n23.84\n26.23\n17.95\n40.68\n19.01\n33.72\n8.54\n13.46\n32.24\n23.52\n29.72\n49.77\n52.44\n40.93\n44.86\n13.2\n37.43\n29.87\n56.61\n12.46\n23.79\n13.29\n39.42\n46.23\n44.52\n23.74\n26.14\n15.52\n43.57\n35.86\n41.05\n28.99\n46.24\n26.92\n10.54\n25.1\n29.07\n9.74\n33.8\n39.84\n26.97\n27.23\n30.65\n33.05\n24.58\n21.91\n30.88\n15.34\n24.34\n23.89\n22.93\n29.41\n28.63\n36.8\n18.29\n32.72\n31.42\n28.94\n40.93\n12.18\n25.56\n36.44\n32.96\n23.84\n26.23\n17.96\n38.63\n19.01\n33.72\n8.54\n13.46\n32.25\n23.52\n29.73\n49.77\n52.45\n40.93\n44.87\n13.2\n37.43\n29.87\n56.62\n12.46\n23.79\n13.29\n39.42\n46.23\n44.52\n23.74\n26.15\n15.53\n43.58\n35.87\n41.05\n28.99\n46.25\n26.92\n10.54\n25.1\n29.07\n9.74\n33.8\n37.17\n33.76\n16.5\n19.99\n36.35\n38.22\n15.42\n33.42\n39.06\n27.68\n26.86\n45.3\n30.12\n15.57\n44.61\n53.52\n57.22\n65.91\n52.83\n33.4\n18.03\n37.36\n35.31\n42.64\n40.06\n43.8\n61.24\n40.87\n33.31\n52.43\n15.09\n38.46\n37.27\n35.23\n42.14\n31.88\n41.54\n39.46\n37.92\n44.28\n31.18\n23.7\n32.77\n32.4"
    },
    {
      "name": ".0/bin1/stack2",
      "format": {
        "type": "csv",
        "parse": {
          "xmin_": "number",
          "xmax_": "number",
          "stack_upr_": "number",
          "stack_lwr_": "number"
        }
      },
      "values": "\"xmin_\",\"xmax_\",\"stack_upr_\",\"stack_lwr_\"\n1,3,1,0\n3,5,5,0\n5,7,7,0\n7,9,14,0\n9,11,25,0\n11,13,28,0\n13,15,37,0\n15,17,31,0\n17,19,33,0\n19,21,26,0\n21,23,37,0\n23,25,51,0\n25,27,49,0\n27,29,32,0\n29,31,37,0\n31,33,55,0\n33,35,55,0\n35,37,42,0\n37,39,53,0\n39,41,57,0\n41,43,43,0\n43,45,42,0\n45,47,25,0\n47,49,24,0\n49,51,23,0\n51,53,30,0\n53,55,21,0\n55,57,34,0\n57,59,11,0\n59,61,15,0\n61,63,13,0\n63,65,10,0\n65,67,15,0\n67,69,10,0\n69,71,4,0\n71,73,10,0\n73,75,8,0\n75,77,3,0\n77,79,5,0\n79,81,7,0\n81,83,2,0"
    },
    {
      "name": "scale/x",
      "format": {
        "type": "csv",
        "parse": {
          "domain": "number"
        }
      },
      "values": "\"domain\"\n-3.1\n87.1"
    },
    {
      "name": "scale/y",
      "format": {
        "type": "csv",
        "parse": {
          "domain": "number"
        }
      },
      "values": "\"domain\"\n0\n86.73"
    }
  ],
  "scales": [
    {
      "name": "x",
      "domain": {
        "data": "scale/x",
        "field": "data.domain"
      },
      "zero": false,
      "nice": false,
      "clamp": false,
      "range": "width"
    },
    {
      "name": "y",
      "domain": {
        "data": "scale/y",
        "field": "data.domain"
      },
      "zero": false,
      "nice": false,
      "clamp": false,
      "range": "height"
    }
  ],
  "marks": [
    {
      "type": "rect",
      "properties": {
        "update": {
          "stroke": {
            "value": "#000000"
          },
          "fill": {
            "value": "#27bc9c"
          },
          "x": {
            "scale": "x",
            "field": "data.xmin_"
          },
          "x2": {
            "scale": "x",
            "field": "data.xmax_"
          },
          "y": {
            "scale": "y",
            "field": "data.stack_upr_"
          },
          "y2": {
            "scale": "y",
            "field": "data.stack_lwr_"
          }
        },
        "ggvis": {
          "data": {
            "value": ".0/bin1/stack2"
          }
        }
      },
      "from": {
        "data": ".0/bin1/stack2"
      }
    },
    {
      "type": "line",
      "properties": {
        "update": {
          "fill": {
            "value": "#27bc9c"
          },
          "y": {
            "scale": "y",
            "field": "data.strength"
          },
          "stroke": {
            "value": "red"
          },
          "x": {
            "scale": "x",
            "value": 35.82
          }
        },
        "ggvis": {
          "data": {
            "value": ".0"
          }
        }
      },
      "from": {
        "data": ".0"
      }
    }
  ],
  "legends": [],
  "axes": [
    {
      "type": "x",
      "scale": "x",
      "orient": "bottom",
      "layer": "back",
      "grid": true,
      "title": "strength"
    },
    {
      "type": "y",
      "scale": "y",
      "orient": "left",
      "layer": "back",
      "grid": true,
      "title": "strength"
    }
  ],
  "padding": null,
  "ggvis_opts": {
    "keep_aspect": false,
    "resizable": true,
    "padding": {},
    "duration": 250,
    "renderer": "svg",
    "hover_duration": 0,
    "width": 672,
    "height": 480
  },
  "handlers": null
};
ggvis.getPlot("plot_id285353667").parseSpec(plot_id285353667_spec);
</script>
<!--/html_preserve-->
From the histogram, it is clear that the distribution is slightly positively skewed. Despite that, there are still majority lot of concretes with strength close to the mean of 35.82. Not too many concretes have strength too strong or weak.

### Scatterplot Matrices

For a general overview of correlation and scatterplots, a scatterplot matrix is plotted for all features.

#### Scatterplot Matrix 1

``` r
pairs.panels(concrete[c("cement", "slag", "ash", "strength")])
```

![](Concrete_Strength_files/figure-markdown_github/unnamed-chunk-5-1.png)

#### Scatterplot Matrix 2

``` r
pairs.panels(concrete[c("superplastic", "coarseagg", "fineagg", "age", "strength")])
```

![](Concrete_Strength_files/figure-markdown_github/unnamed-chunk-6-1.png)

<br />

Step 2: Data Preprocessing & Preparation
========================================

As activation functions in ANNs are sensitive to the change in x values over a small range, the data needs to be normalized. Neural networks work best when the input data are scaled to a **narrow range around zero.**

Traditional Normalization
-------------------------

As most of the feature's and target variable are not normally distributed as seen from the **scatterplot matrices**, traditional normalization instead of Z-Score is more appropriate. Normalization is done using a customized `normalize()` function.

``` r
normalize <- function(x){
  return ((x - min(x))/(max(x) - min(x) ))
}

concrete_norm <- as.data.frame(lapply(concrete, normalize))
```

A preview of the normalized concrete dataset:

``` r
kable(round(head(concrete_norm), digits = 3), caption = "Normalized Data Preview")
```

|  cement|   slag|  ash|  water|  superplastic|  coarseagg|  fineagg|    age|  strength|
|-------:|------:|----:|------:|-------------:|----------:|--------:|------:|---------:|
|   1.000|  0.000|    0|  0.321|         0.078|      0.695|    0.206|  0.074|     0.967|
|   1.000|  0.000|    0|  0.321|         0.078|      0.738|    0.206|  0.074|     0.742|
|   0.526|  0.396|    0|  0.848|         0.000|      0.381|    0.000|  0.739|     0.473|
|   0.526|  0.396|    0|  0.848|         0.000|      0.381|    0.000|  1.000|     0.482|
|   0.221|  0.368|    0|  0.561|         0.000|      0.516|    0.581|  0.986|     0.523|
|   0.374|  0.317|    0|  0.848|         0.000|      0.381|    0.191|  0.245|     0.557|

<br />

Data Preparation
----------------

After normalization, the dataset is ready to be split into its training set and test set. The proportion used here will be 75% training and 25% test. Note that the dataset is already randomly sorted. Therefore, there is no need for random sampling before preparation.

``` r
#training set
concrete_train <- concrete_norm[1:773, ]

#test set
concrete_test <- concrete_norm[774:1030, ]
```

The training set will be used to build the **neural network** and the test set will be used to evaluate how well model generalizes **future data**.

<br />

Step 3: Model Training
======================

The model will be trained using the `neuralnet` package. The package also offers visualization of the network architecture. A multilayer feedforward network with **one hidden node** is constructured. The training is implemented as follows:

``` r
#Build a neural network with one hidden layer 
concrete_model <- neuralnet(strength ~ cement + slag + ash + water + superplastic + coarseagg + fineagg + age , data = concrete_train, hidden = 1)
```

Visualizing Neural Network
--------------------------

The constructed neural network can be visualized simply by using `plot()`

``` r
plot(concrete_model)
```

![](neural.png)

The error shown on the plot is the **Sum of Squared Errors (SSE)**. The lower the SSE the better.

<br />

Step 4: Model Evaluation
========================

As the constructed neural network has trained on the training data, it is now time to put the test data to the test. Note that `compute()` is being used here rather than `predict()`

``` r
#building the predictor, exclude the target variable column
model_results <- compute(concrete_model, concrete_test[1:8])

#store the net.results column 
predicted_strength <- model_results$net.result
```

Model Accuracy
--------------

As this a numeric prediction problem,**correlation** insead of a **confusion matrix** is used to provide insights of the linear association between them both.

``` r
cor(predicted_strength, concrete_test$strength)
```

    ##              [,1]
    ## [1,] 0.7242910187

<br />

Step 5: Improving the Model
===========================

Neural networks with more topology are capable of learning more complex relationships. Hence, **5 hidden nodes** shall be set in the constructed hidden layer in hopes to improve the model.

``` r
#building the new model
concrete_model2 <- neuralnet(strength ~ cement + slag + ash + water + superplastic + coarseagg + fineagg + age, data = concrete_train, hidden = 5 )
```

Visualizing the Improved Neural Network
---------------------------------------

![](neural2.png)

The **SSE** has reduced significantly from 5.67 to only 1.64 with increased in few thousands of steps.

Implementing the Improved Neural Network
----------------------------------------

``` r
#nuilding the new predictor
model_results2 <- compute(concrete_model2, concrete_test[1:8])

#storing the results
predicted_strength2 <- model_results2$net.result
```

Evaluating New Model
--------------------

``` r
cor(predicted_strength2, concrete_test$strength)
```

    ##              [,1]
    ## [1,] 0.7215397756
