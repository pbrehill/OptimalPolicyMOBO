find_frontier_area <- function(data, n, origin_x, origin_y) {

  # Slice top n
  data <- head(data, n)
  y_int <- c(origin_x, max(data$a_mean))
  x_int <- c(max(data$b_mean), origin_y)
  
  
  
  points <- data %>%
    slice(chull(data)) %>%
    arrange(b_mean) %>%
    rbind(y_int, .) %>%
    mutate(width = b_mean - lag(b_mean),
           height = a_mean - origin_y,
           area = width * height)
  
  if(sum(points$area, na.rm = T) < 0) {
    print(head(points))
  }
  
  sum(points$area, na.rm = T)
}

origin_x <- min(test$b_mean)
origin_y <- min(test$a_mean)
bs_effect <- map(test$replicates %>% unique, function (x) {

  test_data <- test %>%
    filter(replicates == x) %>%
    ungroup %>%
    select(b_mean, a_mean) %>%
    mutate(rnum = row_number())

  map_dbl(1:nrow(test_data), ~find_frontier_area(test_data, .x, origin_x, origin_y))
})