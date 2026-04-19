package valkey

import (
	"context"
	"errors"
	"strconv"
	"strings"
	"time"

	"github.com/redis/go-redis/v9"
)

type Client struct {
	ctx context.Context
	rdb *redis.Client
}

func New(addr string, password string, db int) *Client {
	return &Client{
		ctx: context.Background(),
		rdb: redis.NewClient(&redis.Options{
			Addr:     addr,
			Password: password,
			DB:       db,
		}),
	}
}

func (c *Client) Ping() error {
	return c.rdb.Ping(c.ctx).Err()
}

func (c *Client) Close() error {
	return c.rdb.Close()
}

func (c *Client) CurrentModelVersion(datasetID string) (string, error) {
	key := "model:version:" + datasetID
	value, err := c.rdb.Get(c.ctx, key).Result()
	if errors.Is(err, redis.Nil) {
		return "1", nil
	}
	if err != nil {
		return "", err
	}
	if value == "" {
		return "1", nil
	}
	return value, nil
}

func (c *Client) ModelClass(datasetID string) (string, error) {
	key := "model:class:" + datasetID
	value, err := c.rdb.Get(c.ctx, key).Result()
	if errors.Is(err, redis.Nil) {
		return "NaiveForecaster", nil
	}
	if err != nil {
		return "", err
	}
	if value == "" {
		return "NaiveForecaster", nil
	}
	return value, nil
}

func (c *Client) GetCachedPrediction(cacheKey string) (string, bool, error) {
	value, err := c.rdb.Get(c.ctx, cacheKey).Result()
	if errors.Is(err, redis.Nil) {
		return "", false, nil
	}
	if err != nil {
		return "", false, err
	}
	return value, true, nil
}

func (c *Client) SetCachedPrediction(cacheKey string, value string, ttl time.Duration) error {
	return c.rdb.Set(c.ctx, cacheKey, value, ttl).Err()
}

func (c *Client) EnqueueForecastJob(jobID string, datasetID string, fh []int, frequency string, correlationID string) error {
	fhValues := make([]string, len(fh))
	for i, step := range fh {
		fhValues[i] = strconv.Itoa(step)
	}

	return c.rdb.XAdd(c.ctx, &redis.XAddArgs{
		Stream: "forecast:jobs",
		Values: map[string]interface{}{
			"job_id":         jobID,
			"dataset_id":     datasetID,
			"fh":             strings.Join(fhValues, ","),
			"frequency":      frequency,
			"correlation_id": correlationID,
		},
	}).Err()
}

func (c *Client) WaitForResult(correlationID string, timeout time.Duration) (string, error) {
	result, err := c.rdb.BLPop(c.ctx, timeout, "result:"+correlationID).Result()
	if err != nil {
		return "", err
	}
	if len(result) < 2 {
		return "", errors.New("invalid result payload")
	}
	return result[1], nil
}

func (c *Client) TriggerManualRetrain(datasetID string, reason string) error {
	if reason == "" {
		reason = "manual"
	}

	return c.rdb.XAdd(c.ctx, &redis.XAddArgs{
		Stream: "retrain:jobs",
		Values: map[string]interface{}{
			"dataset_id":   datasetID,
			"reason":       reason,
			"triggered_at": time.Now().UTC().Format(time.RFC3339),
		},
	}).Err()
}
