# ===========================================================================
# Per-env ALB wiring on the SHARED public ALB. Two target groups (backend :8000,
# frontend :3000) and two host-header listener rules on the common :443 listener:
#   {uat-}api.<domain>  -> this env's backend
#   {uat-}app.<domain>  -> this env's frontend
# The env's rule priorities live in a band unique to the env (var.listener_rule_
# base_priority) so uat and prod never collide on the shared listener.
# ===========================================================================

resource "aws_lb_target_group" "backend" {
  name        = "${local.name_prefix}-be"
  port        = 8000
  protocol    = "HTTP"
  vpc_id      = var.vpc_id
  target_type = "ip" # Fargate awsvpc
  health_check {
    path                = "/health"
    matcher             = "200"
    interval            = 30
    timeout             = 5
    healthy_threshold   = 2
    unhealthy_threshold = 5
  }
}

resource "aws_lb_target_group" "frontend" {
  name        = "${local.name_prefix}-fe"
  port        = 3000
  protocol    = "HTTP"
  vpc_id      = var.vpc_id
  target_type = "ip"
  health_check {
    path                = "/login"
    matcher             = "200"
    interval            = 30
    timeout             = 5
    healthy_threshold   = 2
    unhealthy_threshold = 5
  }
}

# api.<host> -> backend
resource "aws_lb_listener_rule" "backend" {
  listener_arn = var.https_listener_arn
  priority     = var.listener_rule_base_priority
  action {
    type             = "forward"
    target_group_arn = aws_lb_target_group.backend.arn
  }
  condition {
    host_header {
      values = [var.api_host]
    }
  }
}

# app.<host> -> frontend
resource "aws_lb_listener_rule" "frontend" {
  listener_arn = var.https_listener_arn
  priority     = var.listener_rule_base_priority + 1
  action {
    type             = "forward"
    target_group_arn = aws_lb_target_group.frontend.arn
  }
  condition {
    host_header {
      values = [var.app_host]
    }
  }
}
