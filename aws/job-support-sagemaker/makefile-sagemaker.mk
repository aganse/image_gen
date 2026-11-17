# SageMaker Training Jobs support makefile.
# Included from root Makefile; ECR/build targets are in job-support-common/makefile-common.mk.
#
# Environment variables expected to exist before calling these macros:
#   AWS_ACCT_ID   - AWS account ID
#   AWS_REGION    - AWS region (e.g. us-west-2)
#   SM_SUBNET     - VPC subnet ID for SageMaker compute
#   SM_SG         - Security group ID for SageMaker compute
#   IMAGES_PATH   - S3 URI of training data (e.g. s3://mybucket/afhq)
#   WEIGHTS_PATH  - S3 URI for model weights persistence (e.g. s3://mybucket/weights, optional)
#   MLFLOW_TRACKING_URI - MLflow server URI (e.g. http://10.0.1.50:5000)


# Repo-specific, non-sensitive settings — adjust to taste:
# Override with e.g. make sm-submit SM_INSTANCE_TYPE=ml.g4dn.2xlarge
SM_INSTANCE_TYPE ?= ml.g4dn.xlarge
SM_VOLUME_SIZE=50
SM_MAX_RUNTIME=86400
SM_MAX_WAIT ?= 90000
# Prefix for SageMaker job names; alphanumeric only (no hyphens or underscores) —
# hyphens are used as separators between prefix, script name, instance type, and
# date in the full name.  Derived from PROJECT_NAME (set in root makefile) with
# underscores removed; override in root makefile to taste.  Note SCRIPT names
# must also be alphanumeric since they become part of the job name.
SM_JOB_PREFIX ?= $(subst _,,$(PROJECT_NAME))
# ECR image tag to use; override with e.g. make sm-submit SCRIPT=name TAG=cleanup-gpu
TAG ?= latest

# Spot instance flags: set when SPOT=1 is passed on the command line.
# SM_MAX_WAIT is the total wall-clock deadline including interruption waits
# (must be >= SM_MAX_RUNTIME); override with e.g. make sm-submit SPOT=1 SM_MAX_WAIT=36000
ifdef SPOT
# (recursive '=' rather than ':=' so that JOB_NAME, which is only set inside the
# sm-submit recipe, is expanded at use time and each job gets its own checkpoint prefix)
SM_SPOT_FLAGS = --enable-managed-spot-training \
  --checkpoint-config S3Uri=$(WEIGHTS_PATH)/checkpoints/$(JOB_NAME)/
SM_STOPPING := --stopping-condition \
  MaxRuntimeInSeconds=$(SM_MAX_RUNTIME),MaxWaitTimeInSeconds=$(SM_MAX_WAIT)
else
SM_SPOT_FLAGS :=
SM_STOPPING := --stopping-condition MaxRuntimeInSeconds=$(SM_MAX_RUNTIME)
endif


# Assembled from env vars (sensitive seeds come from environment):
SM_ROLE_ARN=arn:aws:iam::${AWS_ACCT_ID}:role/SageMakerExecutionRole


sm-help:
	@echo "once/rarely:   create-training-subnet VPC_ID=vpc-xxx  create-training-sg VPC_ID=vpc-xxx  (in common)"
	@echo "               sm-create-role  sm-list-role"
	@echo "sometimes:     build  (shared CodeBuild pipeline, pushes to ECR)"
	@echo "image checks:  list-ecr-images"
	@echo "per-run:       create-vpc-endpoints VPC_ID=vpc-xxx  (before)  delete-vpc-endpoints  (after)"
	@echo "more often:    sm-submit SCRIPT=name [TAG=main-gpu]  (TAG defaults to latest)"
	@echo "job checks:    sm-list,  sm-status JOB=name,  sm-logs JOB=name,  sm-cancel JOB=name"

sm-create-role:
	# Create the SageMaker execution role (one-time/rare run).
	# Grants SageMaker permission to pull from ECR, read/write S3, write CloudWatch logs,
	# and access VPC resources.
	@aws iam create-role --role-name SageMakerExecutionRole --no-cli-pager \
		--assume-role-policy-document '{"Version":"2012-10-17","Statement":[{"Effect":"Allow","Principal":{"Service":"sagemaker.amazonaws.com"},"Action":"sts:AssumeRole"}]}'
	@aws iam attach-role-policy --role-name SageMakerExecutionRole \
		--policy-arn arn:aws:iam::aws:policy/AmazonSageMakerFullAccess
	@aws iam attach-role-policy --role-name SageMakerExecutionRole \
		--policy-arn arn:aws:iam::aws:policy/AmazonS3FullAccess
	@echo "SageMakerExecutionRole created."

sm-list-role:
	# Show whether SageMakerExecutionRole exists and its ARN/creation date.
	@aws iam get-role --role-name SageMakerExecutionRole --no-cli-pager \
	  --query '{RoleName:Role.RoleName,Created:Role.CreateDate,Arn:Role.Arn}' \
	  --output table 2>/dev/null || echo "SageMakerExecutionRole does not exist."

sm-submit:
	# Submit a SageMaker Training Job for train_<name>.py with params/params_<name>.json.
	# Usage: make sm-submit SCRIPT=name
ifndef SCRIPT
	@echo "Usage: make sm-submit SCRIPT=name  # runs train_<name>.py with params/params_<name>.json"
	@echo
	@exit 1
endif
ifndef MLFLOW_TRACKING_URI
	$(error MLFLOW_TRACKING_URI is not set. You must export it before running sm-submit)
endif
	@test -f params/params_$(SCRIPT).json || { echo "Error: params/params_$(SCRIPT).json not found" >&2; exit 1; }
	@if [ -z "$(filter $(SCRIPT),$(NO_IMAGES_SCRIPTS))" ] && ! echo "$(IMAGES_PATH)" | grep -q "^s3://"; then \
	  echo "Error: IMAGES_PATH must be an s3:// URI for SCRIPT=$(SCRIPT) SageMaker runs with images" >&2; exit 1; fi
	@if [ -n "$(WEIGHTS_PATH)" ] && ! echo "$(WEIGHTS_PATH)" | grep -q "^s3://"; then \
	  echo "Error: WEIGHTS_PATH must be an s3:// URI if set, got: '$(WEIGHTS_PATH)'" >&2; exit 1; fi
ifndef AWS_ACCT_ID
	$(error AWS_ACCT_ID is not set. Export it before running sm-submit)
endif
ifndef AWS_REGION
	$(error AWS_REGION is not set. Export it before running sm-submit)
endif
ifndef SM_SUBNET
	$(error SM_SUBNET is not set. Export it before running sm-submit)
endif
ifndef SM_SG
	$(error SM_SG is not set. Export it before running sm-submit)
endif
	$(eval INST_ENC := $(shell echo "$(SM_INSTANCE_TYPE)" | sed 's/^ml\.//' | awk -F. '{s=$$2; gsub(/large/,"l",s); sub(/medium/,"m",s); sub(/small/,"s",s); print $$1"-"s}'))
	$(eval JOB_NAME := $(SM_JOB_PREFIX)-$(SCRIPT)-$(INST_ENC)$(if $(SPOT),-spot,)-$(shell date +%Y%m%d-%H%M%S))
	# Params file is minified then base64-encoded (python used for macOS/Linux portability).
	# SageMaker limits each hyperparameter value to 2500 characters, hence the minify + check.
	$(eval PARAMS_BLOB := $(shell python3 -c "import base64,json,sys; print(base64.b64encode(json.dumps(json.load(open(sys.argv[1])),separators=(',',':')).encode()).decode())" params/params_$(SCRIPT).json 2>/dev/null))
	@if [ -z "$(PARAMS_BLOB)" ]; then echo "Error: could not encode params/params_$(SCRIPT).json (invalid JSON?)" >&2; exit 1; fi
	@if [ $$(printf '%s' "$(PARAMS_BLOB)" | wc -c) -gt 2500 ]; then \
	  echo "Error: encoded params/params_$(SCRIPT).json exceeds SageMaker's 2500-char hyperparameter limit; trim the file" >&2; exit 1; fi
	@aws sagemaker create-training-job \
	  --training-job-name $(JOB_NAME) \
	  --algorithm-specification \
	    TrainingImage=$(ECR_REPO_URI):$(TAG),TrainingInputMode=FastFile \
	  --role-arn $(SM_ROLE_ARN) \
	  --input-data-config \
	    '[{"ChannelName":"training","DataSource":{"S3DataSource":{"S3Uri":"${IMAGES_PATH}","S3DataType":"S3Prefix","S3DataDistributionType":"FullyReplicated"}}}]' \
	  --output-data-config S3OutputPath=${WEIGHTS_PATH} \
	  --resource-config \
	    InstanceType=$(SM_INSTANCE_TYPE),InstanceCount=1,VolumeSizeInGB=$(SM_VOLUME_SIZE) \
	  $(SM_SPOT_FLAGS) \
	  $(SM_STOPPING) \
	  --hyper-parameters "{\"params\":\"$(PARAMS_BLOB)\"}" \
	  --environment '{"TRAINING_SCRIPT":"$(SCRIPT)","IMAGE_TAG":"$(TAG)","MLFLOW_TRACKING_URI":"${MLFLOW_TRACKING_URI}","IMAGES_PATH":"/opt/ml/input/data/training","IMAGES_PATH_ORIG":"${IMAGES_PATH}","WEIGHTS_PATH":"${WEIGHTS_PATH}","MLFLOW_ENABLE_ASYNC_LOGGING":"true","JOB_NAME":"$(JOB_NAME)","HOST_USER":"$(shell whoami)"}' \
	  --vpc-config Subnets=${SM_SUBNET},SecurityGroupIds=${SM_SG} \
	  --tags '[{"Key":"JobName","Value":"$(JOB_NAME)"}]' \
	  --no-cli-pager
	@echo "Submitted: $(JOB_NAME)"
	@echo "Check status: make sm-status JOB=$(JOB_NAME)"

sm-list:
	# List recent SageMaker training jobs and their status.
	@aws sagemaker list-training-jobs \
	  --sort-by CreationTime --sort-order Descending \
	  --max-results 20 \
	  --query 'TrainingJobSummaries[*].[CreationTime,TrainingEndTime,TrainingJobStatus,TrainingJobName]' \
	  --output text --no-cli-pager | \
	  python3 aws/job-support-sagemaker/sm-list-fmt.py

sm-status:
	# Show status of a submitted job.  Usage: make sm-status JOB=<name-from-sm-submit>
ifndef JOB
	@echo "Usage: make sm-status JOB=<name-from-sm-submit>  # from output of make sm-submit"
	@echo
	@exit 1
endif
	@aws sagemaker describe-training-job --training-job-name ${JOB} \
	  --query '{Status:TrainingJobStatus,Secondary:SecondaryStatus,Message:SecondaryStatusTransitions[-1].StatusMessage,Reason:FailureReason,InstanceType:ResourceConfig.InstanceType,Start:TrainingStartTime,End:TrainingEndTime}' \
	  --output table --no-cli-pager
	@{ INFO=$$(aws sagemaker describe-training-job --training-job-name ${JOB} --no-cli-pager \
	    --query '[ResourceConfig.InstanceType,TrainingStartTime,TrainingEndTime,EnableManagedSpotTraining]' \
	    --output text 2>/dev/null) && \
	    python3 aws/job-support-sagemaker/sm_cost.py "$(AWS_REGION)" $$INFO; } || true

sm-logs:
	# Tail CloudWatch logs for a running or completed job.
	# Usage: make sm-logs JOB=<name-from-sm-submit>
ifndef JOB
	@echo "Usage: make sm-logs JOB=<name-from-sm-submit>  # from output of make sm-submit"
	@echo
	@exit 1
endif
	@LOG_STREAM=$$(aws logs describe-log-streams \
	  --log-group-name /aws/sagemaker/TrainingJobs \
	  --log-stream-name-prefix ${JOB} \
	  --query 'logStreams[0].logStreamName' --output text 2>/dev/null); \
	if [ "$$LOG_STREAM" = "None" ] || [ -z "$$LOG_STREAM" ]; then \
	  echo "No log stream found yet for job ${JOB}"; \
	else \
	  aws logs get-log-events \
	    --log-group-name /aws/sagemaker/TrainingJobs \
	    --log-stream-name "$$LOG_STREAM" \
	    --no-cli-pager --output json \
	  | python3 -c "import json,sys,re; [sys.stdout.write(re.sub(r'\x1b\[[0-9;]*[A-Za-z]','',e['message']).replace('\r\n','\n').replace('\r','\n')+('' if e['message'].endswith('\n') else '\n')) for e in json.load(sys.stdin)['events']]"; \
	fi

sm-cancel:
	# Stop a running job.  Usage: make sm-cancel JOB=<name-from-sm-submit>
ifndef JOB
	@echo "Usage: make sm-cancel JOB=<name-from-sm-submit>  # from output of make sm-submit"
	@echo
	@exit 1
endif
	@aws sagemaker stop-training-job --training-job-name ${JOB} --no-cli-pager
	@echo "Stop requested for: ${JOB}"


.PHONY: sm-help sm-create-role sm-list-role sm-submit sm-list sm-status sm-logs sm-cancel
