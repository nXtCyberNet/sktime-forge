package config

import (
	"os"
	"strconv"
	"time"
)

type Config struct {
	Port              string
	ValkeyAddr        string
	ValkeyPassword    string
	ValkeyDB          int
	PredictionTimeout time.Duration
}

func Load() Config {
	port := os.Getenv("GATEWAY_PORT")
	if port == "" {
		port = "8080"
	}

	valkeyAddr := os.Getenv("VALKEY_ADDR")
	if valkeyAddr == "" {
		valkeyAddr = "localhost:6379"
	}

	valkeyDB := 0
	if dbStr := os.Getenv("VALKEY_DB"); dbStr != "" {
		if db, err := strconv.Atoi(dbStr); err == nil {
			valkeyDB = db
		}
	}

	return Config{
		Port:              port,
		ValkeyAddr:        valkeyAddr,
		ValkeyPassword:    os.Getenv("VALKEY_PASSWORD"),
		ValkeyDB:          valkeyDB,
		PredictionTimeout: 10 * time.Second,
	}
}
