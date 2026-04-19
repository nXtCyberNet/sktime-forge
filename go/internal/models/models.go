package models

type ForecastRequest struct {
	DatasetID string `json:"dataset" validate:"required"`
	FH        []int  `json:"fh" validate:"required,min=1"`
	Frequency string `json:"frequency"` // Optional
}

type ErrorResponse struct {
	Error string `json:"error"`
}

type ManualRetrainRequest struct {
	DatasetID string `json:"dataset" validate:"required"`
	Reason    string `json:"reason"`
}

type ModelInfoResponse struct {
	DatasetID  string   `json:"dataset"`
	ModelClass string   `json:"model_class"`
	Version    string   `json:"version"`
	CVScore    *float64 `json:"cv_score,omitempty"`
}
