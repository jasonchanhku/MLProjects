My First Machine Learning Project
================

-   Dataset & Objective
    -   Libraries
-   Data Overview
    -   Scatterplots
-   Training and Test Data
    -   Training data:
    -   Test data:
    -   Class Labels:
        -   Training Labels:
        -   Test Labels:
-   Building the Classifer
    -   Comparing the Outcomes
    -   Cross Tabulation Table

Dataset & Objective
===================

I will be using one the most classic dataset, which is iris. This dataset gives the measurements in centimeters of the variables sepal length and width and petal length and width, respectively, for 50 flowers from each of 3 species of iris. The species are **Iris setosa, versicolor, and virginica.**

iris is a data frame with 150 cases (rows) and 5 variables (columns) named **Sepal.Length, Sepal.Width, Petal.Length, Petal.Width, and Species.**

Features used will be **Sepal.Length, Sepal.Width, Petal.Length, Petal.Width** and the target variable is **Species**

Libraries
---------

I imported the following libraries to carry out the analysis

``` r
library(class) # to carry out KNN
library(gmodels) # to check model accuracy
library(ggvis) # for better visualization
```

Data Overview
=============

This is what the iris data.frame table looks like (partial):

    ##   Sepal.Length Sepal.Width Petal.Length Petal.Width Species
    ## 1          5.1         3.5          1.4         0.2  setosa
    ## 2          4.9         3.0          1.4         0.2  setosa
    ## 3          4.7         3.2          1.3         0.2  setosa
    ## 4          4.6         3.1          1.5         0.2  setosa
    ## 5          5.0         3.6          1.4         0.2  setosa
    ## 6          5.4         3.9          1.7         0.4  setosa

Summary statistics of data:

    ##   Sepal.Length    Sepal.Width     Petal.Length    Petal.Width   
    ##  Min.   :4.300   Min.   :2.000   Min.   :1.000   Min.   :0.100  
    ##  1st Qu.:5.100   1st Qu.:2.800   1st Qu.:1.600   1st Qu.:0.300  
    ##  Median :5.800   Median :3.000   Median :4.350   Median :1.300  
    ##  Mean   :5.843   Mean   :3.057   Mean   :3.758   Mean   :1.199  
    ##  3rd Qu.:6.400   3rd Qu.:3.300   3rd Qu.:5.100   3rd Qu.:1.800  
    ##  Max.   :7.900   Max.   :4.400   Max.   :6.900   Max.   :2.500  
    ##        Species  
    ##  setosa    :50  
    ##  versicolor:50  
    ##  virginica :50  
    ##                 
    ##                 
    ## 

Scatterplots
------------

Scatter plots will enable us to see the correlation of the data from a high level <!--html_preserve-->

<nav class="ggvis-control"> <a class="ggvis-dropdown-toggle" title="Controls" onclick="return false;"></a>
<ul class="ggvis-dropdown">
<li>
Renderer: <a id="plot_id942428245_renderer_svg" class="ggvis-renderer-button" onclick="return false;" data-plot-id="plot_id942428245" data-renderer="svg">SVG</a> | <a id="plot_id942428245_renderer_canvas" class="ggvis-renderer-button" onclick="return false;" data-plot-id="plot_id942428245" data-renderer="canvas">Canvas</a>
</li>
<li>
<a id="plot_id942428245_download" class="ggvis-download" data-plot-id="plot_id942428245">Download</a>
</li>
</ul>
</nav>

<script type="text/javascript">
var plot_id942428245_spec = {
  "data": [
    {
      "name": ".0",
      "format": {
        "type": "csv",
        "parse": {
          "Sepal.Length": "number",
          "Sepal.Width": "number"
        }
      },
      "values": "\"Species\",\"Sepal.Length\",\"Sepal.Width\"\n\"setosa\",5.1,3.5\n\"setosa\",4.9,3\n\"setosa\",4.7,3.2\n\"setosa\",4.6,3.1\n\"setosa\",5,3.6\n\"setosa\",5.4,3.9\n\"setosa\",4.6,3.4\n\"setosa\",5,3.4\n\"setosa\",4.4,2.9\n\"setosa\",4.9,3.1\n\"setosa\",5.4,3.7\n\"setosa\",4.8,3.4\n\"setosa\",4.8,3\n\"setosa\",4.3,3\n\"setosa\",5.8,4\n\"setosa\",5.7,4.4\n\"setosa\",5.4,3.9\n\"setosa\",5.1,3.5\n\"setosa\",5.7,3.8\n\"setosa\",5.1,3.8\n\"setosa\",5.4,3.4\n\"setosa\",5.1,3.7\n\"setosa\",4.6,3.6\n\"setosa\",5.1,3.3\n\"setosa\",4.8,3.4\n\"setosa\",5,3\n\"setosa\",5,3.4\n\"setosa\",5.2,3.5\n\"setosa\",5.2,3.4\n\"setosa\",4.7,3.2\n\"setosa\",4.8,3.1\n\"setosa\",5.4,3.4\n\"setosa\",5.2,4.1\n\"setosa\",5.5,4.2\n\"setosa\",4.9,3.1\n\"setosa\",5,3.2\n\"setosa\",5.5,3.5\n\"setosa\",4.9,3.6\n\"setosa\",4.4,3\n\"setosa\",5.1,3.4\n\"setosa\",5,3.5\n\"setosa\",4.5,2.3\n\"setosa\",4.4,3.2\n\"setosa\",5,3.5\n\"setosa\",5.1,3.8\n\"setosa\",4.8,3\n\"setosa\",5.1,3.8\n\"setosa\",4.6,3.2\n\"setosa\",5.3,3.7\n\"setosa\",5,3.3\n\"versicolor\",7,3.2\n\"versicolor\",6.4,3.2\n\"versicolor\",6.9,3.1\n\"versicolor\",5.5,2.3\n\"versicolor\",6.5,2.8\n\"versicolor\",5.7,2.8\n\"versicolor\",6.3,3.3\n\"versicolor\",4.9,2.4\n\"versicolor\",6.6,2.9\n\"versicolor\",5.2,2.7\n\"versicolor\",5,2\n\"versicolor\",5.9,3\n\"versicolor\",6,2.2\n\"versicolor\",6.1,2.9\n\"versicolor\",5.6,2.9\n\"versicolor\",6.7,3.1\n\"versicolor\",5.6,3\n\"versicolor\",5.8,2.7\n\"versicolor\",6.2,2.2\n\"versicolor\",5.6,2.5\n\"versicolor\",5.9,3.2\n\"versicolor\",6.1,2.8\n\"versicolor\",6.3,2.5\n\"versicolor\",6.1,2.8\n\"versicolor\",6.4,2.9\n\"versicolor\",6.6,3\n\"versicolor\",6.8,2.8\n\"versicolor\",6.7,3\n\"versicolor\",6,2.9\n\"versicolor\",5.7,2.6\n\"versicolor\",5.5,2.4\n\"versicolor\",5.5,2.4\n\"versicolor\",5.8,2.7\n\"versicolor\",6,2.7\n\"versicolor\",5.4,3\n\"versicolor\",6,3.4\n\"versicolor\",6.7,3.1\n\"versicolor\",6.3,2.3\n\"versicolor\",5.6,3\n\"versicolor\",5.5,2.5\n\"versicolor\",5.5,2.6\n\"versicolor\",6.1,3\n\"versicolor\",5.8,2.6\n\"versicolor\",5,2.3\n\"versicolor\",5.6,2.7\n\"versicolor\",5.7,3\n\"versicolor\",5.7,2.9\n\"versicolor\",6.2,2.9\n\"versicolor\",5.1,2.5\n\"versicolor\",5.7,2.8\n\"virginica\",6.3,3.3\n\"virginica\",5.8,2.7\n\"virginica\",7.1,3\n\"virginica\",6.3,2.9\n\"virginica\",6.5,3\n\"virginica\",7.6,3\n\"virginica\",4.9,2.5\n\"virginica\",7.3,2.9\n\"virginica\",6.7,2.5\n\"virginica\",7.2,3.6\n\"virginica\",6.5,3.2\n\"virginica\",6.4,2.7\n\"virginica\",6.8,3\n\"virginica\",5.7,2.5\n\"virginica\",5.8,2.8\n\"virginica\",6.4,3.2\n\"virginica\",6.5,3\n\"virginica\",7.7,3.8\n\"virginica\",7.7,2.6\n\"virginica\",6,2.2\n\"virginica\",6.9,3.2\n\"virginica\",5.6,2.8\n\"virginica\",7.7,2.8\n\"virginica\",6.3,2.7\n\"virginica\",6.7,3.3\n\"virginica\",7.2,3.2\n\"virginica\",6.2,2.8\n\"virginica\",6.1,3\n\"virginica\",6.4,2.8\n\"virginica\",7.2,3\n\"virginica\",7.4,2.8\n\"virginica\",7.9,3.8\n\"virginica\",6.4,2.8\n\"virginica\",6.3,2.8\n\"virginica\",6.1,2.6\n\"virginica\",7.7,3\n\"virginica\",6.3,3.4\n\"virginica\",6.4,3.1\n\"virginica\",6,3\n\"virginica\",6.9,3.1\n\"virginica\",6.7,3.1\n\"virginica\",6.9,3.1\n\"virginica\",5.8,2.7\n\"virginica\",6.8,3.2\n\"virginica\",6.7,3.3\n\"virginica\",6.7,3\n\"virginica\",6.3,2.5\n\"virginica\",6.5,3\n\"virginica\",6.2,3.4\n\"virginica\",5.9,3"
    },
    {
      "name": "scale/fill",
      "format": {
        "type": "csv",
        "parse": {}
      },
      "values": "\"domain\"\n\"setosa\"\n\"versicolor\"\n\"virginica\""
    },
    {
      "name": "scale/x",
      "format": {
        "type": "csv",
        "parse": {
          "domain": "number"
        }
      },
      "values": "\"domain\"\n4.12\n8.08"
    },
    {
      "name": "scale/y",
      "format": {
        "type": "csv",
        "parse": {
          "domain": "number"
        }
      },
      "values": "\"domain\"\n1.88\n4.52"
    }
  ],
  "scales": [
    {
      "name": "fill",
      "type": "ordinal",
      "domain": {
        "data": "scale/fill",
        "field": "data.domain"
      },
      "points": true,
      "sort": false,
      "range": "category10"
    },
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
      "type": "symbol",
      "properties": {
        "update": {
          "size": {
            "value": 50
          },
          "fill": {
            "scale": "fill",
            "field": "data.Species"
          },
          "x": {
            "scale": "x",
            "field": "data.Sepal\\.Length"
          },
          "y": {
            "scale": "y",
            "field": "data.Sepal\\.Width"
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
  "legends": [
    {
      "orient": "right",
      "fill": "fill",
      "title": "Species"
    }
  ],
  "axes": [
    {
      "type": "x",
      "scale": "x",
      "orient": "bottom",
      "layer": "back",
      "grid": true,
      "title": "Sepal.Length"
    },
    {
      "type": "y",
      "scale": "y",
      "orient": "left",
      "layer": "back",
      "grid": true,
      "title": "Sepal.Width"
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
ggvis.getPlot("plot_id942428245").parseSpec(plot_id942428245_spec);
</script>
<!--/html_preserve-->
There is a rather high positive correlation for setosa compared to versicolor and virginica when it comes to its sepal length and width.

<!--html_preserve-->

<nav class="ggvis-control"> <a class="ggvis-dropdown-toggle" title="Controls" onclick="return false;"></a>
<ul class="ggvis-dropdown">
<li>
Renderer: <a id="plot_id338462363_renderer_svg" class="ggvis-renderer-button" onclick="return false;" data-plot-id="plot_id338462363" data-renderer="svg">SVG</a> | <a id="plot_id338462363_renderer_canvas" class="ggvis-renderer-button" onclick="return false;" data-plot-id="plot_id338462363" data-renderer="canvas">Canvas</a>
</li>
<li>
<a id="plot_id338462363_download" class="ggvis-download" data-plot-id="plot_id338462363">Download</a>
</li>
</ul>
</nav>

<script type="text/javascript">
var plot_id338462363_spec = {
  "data": [
    {
      "name": ".0",
      "format": {
        "type": "csv",
        "parse": {
          "Petal.Length": "number",
          "Petal.Width": "number"
        }
      },
      "values": "\"Species\",\"Petal.Length\",\"Petal.Width\"\n\"setosa\",1.4,0.2\n\"setosa\",1.4,0.2\n\"setosa\",1.3,0.2\n\"setosa\",1.5,0.2\n\"setosa\",1.4,0.2\n\"setosa\",1.7,0.4\n\"setosa\",1.4,0.3\n\"setosa\",1.5,0.2\n\"setosa\",1.4,0.2\n\"setosa\",1.5,0.1\n\"setosa\",1.5,0.2\n\"setosa\",1.6,0.2\n\"setosa\",1.4,0.1\n\"setosa\",1.1,0.1\n\"setosa\",1.2,0.2\n\"setosa\",1.5,0.4\n\"setosa\",1.3,0.4\n\"setosa\",1.4,0.3\n\"setosa\",1.7,0.3\n\"setosa\",1.5,0.3\n\"setosa\",1.7,0.2\n\"setosa\",1.5,0.4\n\"setosa\",1,0.2\n\"setosa\",1.7,0.5\n\"setosa\",1.9,0.2\n\"setosa\",1.6,0.2\n\"setosa\",1.6,0.4\n\"setosa\",1.5,0.2\n\"setosa\",1.4,0.2\n\"setosa\",1.6,0.2\n\"setosa\",1.6,0.2\n\"setosa\",1.5,0.4\n\"setosa\",1.5,0.1\n\"setosa\",1.4,0.2\n\"setosa\",1.5,0.2\n\"setosa\",1.2,0.2\n\"setosa\",1.3,0.2\n\"setosa\",1.4,0.1\n\"setosa\",1.3,0.2\n\"setosa\",1.5,0.2\n\"setosa\",1.3,0.3\n\"setosa\",1.3,0.3\n\"setosa\",1.3,0.2\n\"setosa\",1.6,0.6\n\"setosa\",1.9,0.4\n\"setosa\",1.4,0.3\n\"setosa\",1.6,0.2\n\"setosa\",1.4,0.2\n\"setosa\",1.5,0.2\n\"setosa\",1.4,0.2\n\"versicolor\",4.7,1.4\n\"versicolor\",4.5,1.5\n\"versicolor\",4.9,1.5\n\"versicolor\",4,1.3\n\"versicolor\",4.6,1.5\n\"versicolor\",4.5,1.3\n\"versicolor\",4.7,1.6\n\"versicolor\",3.3,1\n\"versicolor\",4.6,1.3\n\"versicolor\",3.9,1.4\n\"versicolor\",3.5,1\n\"versicolor\",4.2,1.5\n\"versicolor\",4,1\n\"versicolor\",4.7,1.4\n\"versicolor\",3.6,1.3\n\"versicolor\",4.4,1.4\n\"versicolor\",4.5,1.5\n\"versicolor\",4.1,1\n\"versicolor\",4.5,1.5\n\"versicolor\",3.9,1.1\n\"versicolor\",4.8,1.8\n\"versicolor\",4,1.3\n\"versicolor\",4.9,1.5\n\"versicolor\",4.7,1.2\n\"versicolor\",4.3,1.3\n\"versicolor\",4.4,1.4\n\"versicolor\",4.8,1.4\n\"versicolor\",5,1.7\n\"versicolor\",4.5,1.5\n\"versicolor\",3.5,1\n\"versicolor\",3.8,1.1\n\"versicolor\",3.7,1\n\"versicolor\",3.9,1.2\n\"versicolor\",5.1,1.6\n\"versicolor\",4.5,1.5\n\"versicolor\",4.5,1.6\n\"versicolor\",4.7,1.5\n\"versicolor\",4.4,1.3\n\"versicolor\",4.1,1.3\n\"versicolor\",4,1.3\n\"versicolor\",4.4,1.2\n\"versicolor\",4.6,1.4\n\"versicolor\",4,1.2\n\"versicolor\",3.3,1\n\"versicolor\",4.2,1.3\n\"versicolor\",4.2,1.2\n\"versicolor\",4.2,1.3\n\"versicolor\",4.3,1.3\n\"versicolor\",3,1.1\n\"versicolor\",4.1,1.3\n\"virginica\",6,2.5\n\"virginica\",5.1,1.9\n\"virginica\",5.9,2.1\n\"virginica\",5.6,1.8\n\"virginica\",5.8,2.2\n\"virginica\",6.6,2.1\n\"virginica\",4.5,1.7\n\"virginica\",6.3,1.8\n\"virginica\",5.8,1.8\n\"virginica\",6.1,2.5\n\"virginica\",5.1,2\n\"virginica\",5.3,1.9\n\"virginica\",5.5,2.1\n\"virginica\",5,2\n\"virginica\",5.1,2.4\n\"virginica\",5.3,2.3\n\"virginica\",5.5,1.8\n\"virginica\",6.7,2.2\n\"virginica\",6.9,2.3\n\"virginica\",5,1.5\n\"virginica\",5.7,2.3\n\"virginica\",4.9,2\n\"virginica\",6.7,2\n\"virginica\",4.9,1.8\n\"virginica\",5.7,2.1\n\"virginica\",6,1.8\n\"virginica\",4.8,1.8\n\"virginica\",4.9,1.8\n\"virginica\",5.6,2.1\n\"virginica\",5.8,1.6\n\"virginica\",6.1,1.9\n\"virginica\",6.4,2\n\"virginica\",5.6,2.2\n\"virginica\",5.1,1.5\n\"virginica\",5.6,1.4\n\"virginica\",6.1,2.3\n\"virginica\",5.6,2.4\n\"virginica\",5.5,1.8\n\"virginica\",4.8,1.8\n\"virginica\",5.4,2.1\n\"virginica\",5.6,2.4\n\"virginica\",5.1,2.3\n\"virginica\",5.1,1.9\n\"virginica\",5.9,2.3\n\"virginica\",5.7,2.5\n\"virginica\",5.2,2.3\n\"virginica\",5,1.9\n\"virginica\",5.2,2\n\"virginica\",5.4,2.3\n\"virginica\",5.1,1.8"
    },
    {
      "name": "scale/fill",
      "format": {
        "type": "csv",
        "parse": {}
      },
      "values": "\"domain\"\n\"setosa\"\n\"versicolor\"\n\"virginica\""
    },
    {
      "name": "scale/x",
      "format": {
        "type": "csv",
        "parse": {
          "domain": "number"
        }
      },
      "values": "\"domain\"\n0.705\n7.195"
    },
    {
      "name": "scale/y",
      "format": {
        "type": "csv",
        "parse": {
          "domain": "number"
        }
      },
      "values": "\"domain\"\n-0.02\n2.62"
    }
  ],
  "scales": [
    {
      "name": "fill",
      "type": "ordinal",
      "domain": {
        "data": "scale/fill",
        "field": "data.domain"
      },
      "points": true,
      "sort": false,
      "range": "category10"
    },
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
      "type": "symbol",
      "properties": {
        "update": {
          "size": {
            "value": 50
          },
          "fill": {
            "scale": "fill",
            "field": "data.Species"
          },
          "x": {
            "scale": "x",
            "field": "data.Petal\\.Length"
          },
          "y": {
            "scale": "y",
            "field": "data.Petal\\.Width"
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
  "legends": [
    {
      "orient": "right",
      "fill": "fill",
      "title": "Species"
    }
  ],
  "axes": [
    {
      "type": "x",
      "scale": "x",
      "orient": "bottom",
      "layer": "back",
      "grid": true,
      "title": "Petal.Length"
    },
    {
      "type": "y",
      "scale": "y",
      "orient": "left",
      "layer": "back",
      "grid": true,
      "title": "Petal.Width"
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
ggvis.getPlot("plot_id338462363").parseSpec(plot_id338462363_spec);
</script>
<!--/html_preserve-->
However, for petal length and width, all 3 species have pretty high positive correlation

Training and Test Data
======================

I will use the training set to train the system and the test set to evaluate and test the trained system. The ratio of training to test set i will use is 1:3.

Training data:
--------------

    ##   Sepal.Length Sepal.Width Petal.Length Petal.Width
    ## 1          5.1         3.5          1.4         0.2
    ## 2          4.9         3.0          1.4         0.2
    ## 3          4.7         3.2          1.3         0.2
    ## 4          4.6         3.1          1.5         0.2
    ## 6          5.4         3.9          1.7         0.4
    ## 7          4.6         3.4          1.4         0.3

Test data:
----------

    ##    Sepal.Length Sepal.Width Petal.Length Petal.Width
    ## 5           5.0         3.6          1.4         0.2
    ## 11          5.4         3.7          1.5         0.2
    ## 14          4.3         3.0          1.1         0.1
    ## 16          5.7         4.4          1.5         0.4
    ## 26          5.0         3.0          1.6         0.2
    ## 28          5.2         3.5          1.5         0.2

Class Labels:
-------------

The class labels contain the target variable for the training and test data.

### Training Labels:

    ## [1] setosa setosa setosa setosa setosa setosa
    ## Levels: setosa versicolor virginica

### Test Labels:

    ## [1] setosa setosa setosa setosa setosa setosa
    ## Levels: setosa versicolor virginica

Building the Classifer
======================

The machine learning algorithm i will be using is **K Nearest Neighbour** to classify the test data for our target variable, Species. The parameter k used here is 3.

``` r
iris_pred <- knn(train = iris.training, test = iris.test, cl = iris.trainLabels, k=3)
iris_pred
```

    ##  [1] setosa     setosa     setosa     setosa     setosa     setosa    
    ##  [7] setosa     setosa     setosa     setosa     setosa     setosa    
    ## [13] versicolor versicolor versicolor versicolor versicolor versicolor
    ## [19] versicolor versicolor versicolor versicolor versicolor versicolor
    ## [25] virginica  virginica  virginica  virginica  versicolor virginica 
    ## [31] virginica  virginica  virginica  virginica  virginica  virginica 
    ## [37] virginica  virginica  virginica  virginica 
    ## Levels: setosa versicolor virginica

Comparing the Outcomes
----------------------

Here is where I compare the model's performance on the predicted species to the observed Species.

    ##    iris.testLabels  iris_pred
    ## 1           setosa     setosa
    ## 2           setosa     setosa
    ## 3           setosa     setosa
    ## 4           setosa     setosa
    ## 5           setosa     setosa
    ## 6           setosa     setosa
    ## 7           setosa     setosa
    ## 8           setosa     setosa
    ## 9           setosa     setosa
    ## 10          setosa     setosa
    ## 11          setosa     setosa
    ## 12          setosa     setosa
    ## 13      versicolor versicolor
    ## 14      versicolor versicolor
    ## 15      versicolor versicolor
    ## 16      versicolor versicolor
    ## 17      versicolor versicolor
    ## 18      versicolor versicolor
    ## 19      versicolor versicolor
    ## 20      versicolor versicolor
    ## 21      versicolor versicolor
    ## 22      versicolor versicolor
    ## 23      versicolor versicolor
    ## 24      versicolor versicolor
    ## 25       virginica  virginica
    ## 26       virginica  virginica
    ## 27       virginica  virginica
    ## 28       virginica  virginica
    ## 29       virginica versicolor
    ## 30       virginica  virginica
    ## 31       virginica  virginica
    ## 32       virginica  virginica
    ## 33       virginica  virginica
    ## 34       virginica  virginica
    ## 35       virginica  virginica
    ## 36       virginica  virginica
    ## 37       virginica  virginica
    ## 38       virginica  virginica
    ## 39       virginica  virginica
    ## 40       virginica  virginica

Seems like the model managed to predict everything correct except for one entry on the 29th row.

Cross Tabulation Table
----------------------

Cross tabulation table helps understand the relationship between the observed and predicted species.

    ## 
    ##  
    ##    Cell Contents
    ## |-------------------------|
    ## |                       N |
    ## |           N / Row Total |
    ## |           N / Col Total |
    ## |         N / Table Total |
    ## |-------------------------|
    ## 
    ##  
    ## Total Observations in Table:  40 
    ## 
    ##  
    ##                 | iris_pred 
    ## iris.testLabels |     setosa | versicolor |  virginica |  Row Total | 
    ## ----------------|------------|------------|------------|------------|
    ##          setosa |         12 |          0 |          0 |         12 | 
    ##                 |      1.000 |      0.000 |      0.000 |      0.300 | 
    ##                 |      1.000 |      0.000 |      0.000 |            | 
    ##                 |      0.300 |      0.000 |      0.000 |            | 
    ## ----------------|------------|------------|------------|------------|
    ##      versicolor |          0 |         12 |          0 |         12 | 
    ##                 |      0.000 |      1.000 |      0.000 |      0.300 | 
    ##                 |      0.000 |      0.923 |      0.000 |            | 
    ##                 |      0.000 |      0.300 |      0.000 |            | 
    ## ----------------|------------|------------|------------|------------|
    ##       virginica |          0 |          1 |         15 |         16 | 
    ##                 |      0.000 |      0.062 |      0.938 |      0.400 | 
    ##                 |      0.000 |      0.077 |      1.000 |            | 
    ##                 |      0.000 |      0.025 |      0.375 |            | 
    ## ----------------|------------|------------|------------|------------|
    ##    Column Total |         12 |         13 |         15 |         40 | 
    ##                 |      0.300 |      0.325 |      0.375 |            | 
    ## ----------------|------------|------------|------------|------------|
    ## 
    ##
