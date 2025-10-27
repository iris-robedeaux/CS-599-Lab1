#import packages
library(tidyverse)

#import all the data
files <- c(
  "~/GitHub/CS-599-Lab1/Assignment3/MLP_hl2_optAdam_lossL1+L2.csv",
  "~/GitHub/CS-599-Lab1/Assignment3/MLP_hl2_optAdam_lossL2.csv",
  "~/GitHub/CS-599-Lab1/Assignment3/MLP_hl2_optAdam_lossNLL.csv",
  "~/GitHub/CS-599-Lab1/Assignment3/MLP_hl2_optAdam_lossNLL_withDropout.csv",
  "~/GitHub/CS-599-Lab1/Assignment3/MLP_hl2_optRMSprop_lossNLL_withDropout.csv",
  "~/GitHub/CS-599-Lab1/Assignment3/MLP_hl2_optSGD_lossNLL_withDropout.csv",
  "~/GitHub/CS-599-Lab1/Assignment3/MLP_hl3_optAdam_lossL1+L2.csv",
  "~/GitHub/CS-599-Lab1/Assignment3/MLP_hl3_optAdam_lossNLL.csv",
  "~/GitHub/CS-599-Lab1/Assignment3/MLP_hl4_optAdam_lossNLL.csv"
)

# Read all files and add metadata from the filename
data_list <- lapply(files, function(f) {
  
  # Read CSV (no header)
  df <- read.csv(f, header = FALSE)
  
  # Extract parts of the filename
  name <- tools::file_path_sans_ext(basename(f))
  
  # Extract attributes with regex
  hidden_layers <- str_extract(name, "(?<=hl)\\d+")
  optimizer     <- str_extract(name, "(?<=opt)[A-Za-z]+")
  loss          <- str_extract(name, "(?<=loss)[\\w+]+")
  dropout       <- str_detect(name, "withDropout")
  
  # Add as a column
  df <- df %>%
    mutate(
      trainTask = seq_len(n()),
      hidden_layers = as.integer(hidden_layers),
      optimizer = optimizer,
      loss = loss,
      dropout = dropout
    )
  
  return(df)
})

# Combine them all into one big dataframe
combined <- bind_rows(data_list)

#replace zeros with NAs (unlearned tasks yet)
combined[] <- lapply(combined, function(x) {
  if (is.numeric(x)) x[x == 0] <- NA
  x
})

#tidydata
combinedTidy <- combined %>%
  pivot_longer(
    cols = starts_with("V"),     # all V1, V2, … columns
    names_to = "taskNum",        # name of the new column
    values_to = "value"          # name of the value column
  )

combinedTidy <- combinedTidy %>%
  mutate(taskNum = as.integer(sub("V", "", taskNum)))

#factorize them
combinedTidy$trainTask <- as.factor(combinedTidy$trainTask)
combinedTidy$taskNum <- as.factor(combinedTidy$taskNum)

#first graph, testing loss functions
lossData <- combinedTidy %>% filter( optimizer == "Adam", hidden_layers == 2, dropout == FALSE, loss != "L2")

lossForget <-
  ggplot( data = subset(lossData, !is.na(value)), 
          aes( x = trainTask, y = value, color = taskNum, 
               group = taskNum) ) +
    geom_point() +
    geom_line() +
    facet_wrap(~ loss, ncol=2) +
    theme_bw() +
    guides(color = guide_legend(ncol = 2)) +
    labs( title = "Effect of Loss on Forgetting",
          x = "Task Trained On",
          y = "Accuracy",
          color = "Task Tested On")

#next graphing layers
layerData <- combinedTidy %>% filter( optimizer == "Adam", loss == "NLL", dropout == FALSE)

ggplot( data = subset(layerData, !is.na(value)), 
          aes( x = trainTask, y = value, color = taskNum, 
               group = taskNum) ) +
  geom_point() +
  geom_line() +
  facet_wrap(~ hidden_layers, ncol=3) +
  theme_bw() +
  guides(color = guide_legend(ncol = 2)) +
  labs( title = "Effect of Layers on Forgetting",
        x = "Task Trained On",
        y = "Accuracy",
        color = "Task Tested On")

#next graphing dropout
dropData <- combinedTidy %>% filter( optimizer == "Adam", hidden_layers == 2, loss == "NLL_withDropout" | loss == "NLL")
dropData <- gender <- fct_recode(dropout, TRUE = "Male", FALSE = "Female")

ggplot( data = subset(dropData, !is.na(value)), 
        aes( x = trainTask, y = value, color = taskNum, 
             group = taskNum) ) +
  geom_point() +
  geom_line() +
  facet_wrap(~ dropout, ncol=2) +
  theme_bw() +
  guides(color = guide_legend(ncol = 2)) +
  labs( title = "Effect of Dropout on Forgetting",
        x = "Task Trained On",
        y = "Accuracy",
        color = "Task Tested On")
