package api

import (
	"crypto/sha1"
	"encoding/hex"
	"fmt"
	"strings"

	"github.com/go-playground/validator/v10"
	"github.com/gofiber/fiber/v2"
	"github.com/google/uuid"

	"sktime-agentic/internal/config"
	"sktime-agentic/internal/models"
	"sktime-agentic/internal/valkey"
)

type Handler struct {
	valkey   *valkey.Client
	validate *validator.Validate
	config   config.Config
}

func NewHandler(client *valkey.Client, cfg config.Config) *Handler {
	return &Handler{
		valkey:   client,
		validate: validator.New(),
		config:   cfg,
	}
}

func (h *Handler) Health(c *fiber.Ctx) error {
	return c.JSON(fiber.Map{"status": "ok"})
}

func (h *Handler) Ready(c *fiber.Ctx) error {
	if err := h.valkey.Ping(); err != nil {
		return c.Status(fiber.StatusServiceUnavailable).JSON(models.ErrorResponse{Error: err.Error()})
	}
	return c.JSON(fiber.Map{"status": "ready"})
}

func (h *Handler) Metrics(c *fiber.Ctx) error {
	return c.Type("text/plain").SendString("sktime_agentic_gateway_up 1\n")
}

func (h *Handler) Forecast(c *fiber.Ctx) error {
	var req models.ForecastRequest
	if err := c.BodyParser(&req); err != nil {
		return c.Status(fiber.StatusBadRequest).JSON(models.ErrorResponse{Error: "invalid request"})
	}
	if err := h.validate.Struct(req); err != nil {
		return c.Status(fiber.StatusUnprocessableEntity).JSON(models.ErrorResponse{Error: err.Error()})
	}

	modelVersion, err := h.valkey.CurrentModelVersion(req.DatasetID)
	if err != nil {
		return c.Status(fiber.StatusServiceUnavailable).JSON(models.ErrorResponse{Error: err.Error()})
	}

	cacheKey := fmt.Sprintf("pred:%s:%s:%s", modelVersion, req.DatasetID, hashFH(req.FH))
	cached, hit, err := h.valkey.GetCachedPrediction(cacheKey)
	if err != nil {
		return c.Status(fiber.StatusServiceUnavailable).JSON(models.ErrorResponse{Error: err.Error()})
	}
	if hit {
		return c.Type("application/json").SendString(cached)
	}

	correlationID := uuid.NewString()
	if err := h.valkey.EnqueueForecastJob(correlationID, req.DatasetID, req.FH, req.Frequency, correlationID); err != nil {
		return c.Status(fiber.StatusServiceUnavailable).JSON(models.ErrorResponse{Error: err.Error()})
	}

	result, err := h.valkey.WaitForResult(correlationID, h.config.PredictionTimeout)
	if err != nil {
		c.Set("Retry-After", "5")
		return c.Status(fiber.StatusServiceUnavailable).JSON(
			models.ErrorResponse{Error: "prediction timeout, please retry"},
		)
	}

	return c.Type("application/json").SendString(result)
}

func (h *Handler) AdminRetrain(c *fiber.Ctx) error {
	var req models.ManualRetrainRequest
	if err := c.BodyParser(&req); err != nil {
		return c.Status(fiber.StatusBadRequest).JSON(models.ErrorResponse{Error: "invalid request"})
	}
	if err := h.validate.Struct(req); err != nil {
		return c.Status(fiber.StatusUnprocessableEntity).JSON(models.ErrorResponse{Error: err.Error()})
	}

	if err := h.valkey.TriggerManualRetrain(req.DatasetID, req.Reason); err != nil {
		return c.Status(fiber.StatusServiceUnavailable).JSON(models.ErrorResponse{Error: err.Error()})
	}

	return c.JSON(fiber.Map{"status": "queued"})
}

func (h *Handler) AdminModel(c *fiber.Ctx) error {
	datasetID := strings.TrimSpace(c.Params("dataset"))
	if datasetID == "" {
		return c.Status(fiber.StatusBadRequest).JSON(models.ErrorResponse{Error: "dataset is required"})
	}

	version, err := h.valkey.CurrentModelVersion(datasetID)
	if err != nil {
		return c.Status(fiber.StatusServiceUnavailable).JSON(models.ErrorResponse{Error: err.Error()})
	}
	modelClass, err := h.valkey.ModelClass(datasetID)
	if err != nil {
		return c.Status(fiber.StatusServiceUnavailable).JSON(models.ErrorResponse{Error: err.Error()})
	}

	return c.JSON(models.ModelInfoResponse{
		DatasetID:  datasetID,
		ModelClass: modelClass,
		Version:    version,
		CVScore:    nil,
	})
}

func hashFH(fh []int) string {
	b := make([]byte, 0, len(fh)*3)
	for i, step := range fh {
		if i > 0 {
			b = append(b, ',')
		}
		b = append(b, []byte(fmt.Sprintf("%d", step))...)
	}
	hash := sha1.Sum(b)
	return hex.EncodeToString(hash[:])[:12]
}
