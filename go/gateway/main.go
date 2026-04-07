package main

import (
	"fmt"
	"log"

	"github.com/gofiber/fiber/v2"

	"sktime-agentic/internal/api"
	"sktime-agentic/internal/config"
	"sktime-agentic/internal/valkey"
)

func main() {
	cfg := config.Load()
	valkeyClient := valkey.New(cfg.ValkeyAddr, cfg.ValkeyPassword, cfg.ValkeyDB)
	defer func() {
		_ = valkeyClient.Close()
	}()

	handler := api.NewHandler(valkeyClient, cfg)

	app := fiber.New()

	app.Get("/health", handler.Health)
	app.Get("/ready", handler.Ready)
	app.Get("/metrics", handler.Metrics)
	app.Post("/forecast", handler.Forecast)
	app.Post("/admin/retrain", handler.AdminRetrain)
	app.Get("/admin/model/:dataset", handler.AdminModel)

	log.Fatal(app.Listen(fmt.Sprintf(":%s", cfg.Port)))
}
