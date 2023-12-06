# Load necessary libraries
library(plotly)
library(ggplot2)
library(sf)
library(rnaturalearth)
library(dplyr)
library(readr)

# Load and prepare your data
players_data <- read_csv("Project/p2/overwatch_league_player_nationalities_updated.csv")
players_data$Representation <- as.numeric(players_data$Representation)
world <- ne_countries(scale = "medium", returnclass = "sf")
world_data <- left_join(world, players_data, by = c("name" = "Country / Region"))

# Create the Plotly plot
p <- ggplot(world_data) +
  geom_sf(aes(fill = Representation), color = NA) +
  scale_fill_viridis_c(name = "Number of Players") +
  theme_void()

plotly_plot <- ggplotly(p, tooltip = "Representation")

print(plotly_plot)