const { createProxyMiddleware } = require("http-proxy-middleware");

const proxyOptions = {
  target: "http://localhost:5000",
  changeOrigin: true,
  proxyTimeout: 120000,
  timeout: 120000,
};

module.exports = function (app) {
  app.use("/upload", createProxyMiddleware(proxyOptions));
  app.use("/recommend", createProxyMiddleware(proxyOptions));
};
