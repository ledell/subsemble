.onAttach <- function(...) {
  packageStartupMessage('subsemble (beta)')
  packageStartupMessage('Version: ', utils::packageDescription('subsemble')$Version)
  packageStartupMessage('Package created on ', utils::packageDescription('subsemble')$Date, '\n')
  packageStartupMessage('Notice to subsemble users:')
  packageStartupMessage('The interface (function arguments/values) may be subject to change prior to version 1.0.0.')
}
