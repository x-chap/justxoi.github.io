library(plotly)
library(sf)
library(rnaturalearth)
library(dplyr)
library(readr)
# Load the dataset
players_data <- read_csv("Project/p2/overwatch_league_player_nationalities_updated.csv") # Replace with your file path

# Convert the Representation column to numeric
players_data$Representation <- as.numeric(players_data$Representation)

# Get world map data
world <- ne_countries(scale = "medium", returnclass = "sf")

# Join your data with the world map data
world_data <- left_join(world, players_data, by = c("name" = "Country / Region"))

# Create the plotly map
p <- ggplot(world_data) +
  geom_sf(aes(fill = Representation), color = NA) +
  scale_fill_viridis_c(name = "Number of Players") +
  theme_void()

plot_map <- ggplotly(p, tooltip = "name")

# Generate a list of countries and their proportions
country_list <- paste(world_data$name, ":", world_data$Representation, sep=" ")
country_list_html <- paste("<ul>", paste("<li>", country_list, "</li>", collapse = ""), "</ul>")

# Print or use the HTML list as needed
cat(country_list_html)
