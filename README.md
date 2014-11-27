# subsemble

The `subsemble` package is an R implementation of the Subsemble algorithm. Subsemble is a general subset ensemble prediction method, which can be used for small, moderate, or large datasets. Subsemble partitions the full dataset into subsets of observations, fits a specified underlying algorithm on each subset, and uses a unique form of V-fold cross-validation to output a prediction function that combines the subset-specific fits. An oracle result provides a theoretical performance guarantee for Subsemble.

Stephanie Sapp, Mark J. van der Laan & John Canny, Journal of Applied Statistics (2013). Subsemble: An ensemble method for combining subset-specific algorithm fits

- Article: [http://www.tandfonline.com/doi/abs/10.1080/02664763.2013.864263](http://www.tandfonline.com/doi/abs/10.1080/02664763.2013.864263)
- Preprint: [https://biostats.bepress.com/ucbbiostat/paper313](https://biostats.bepress.com/ucbbiostat/paper313)

