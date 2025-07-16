import torch
import torch.optim as optim
import random
from gymnasium import spaces

def collect_transition_data(env, data_size):
    """state, action, next_state tuples using random actions."""
    data = []
    state, _ = env.reset()
    for _ in range(data_size):
        action = env.action_space.sample()  # Sample random action
        next_state, reward, done, truncated, _ = env.step(action)
        data.append((state, action, next_state))
        state = next_state
        if done or truncated:
            state, _ = env.reset()
    return data

def train_transition_model(state_dim, action_dim, data, config):
    """Entrena un modelo de transicion"""
    model = TransitionModel(state_dim, action_dim, config["transition_model_hidden_layers"])
    optimizer = optim.Adam(model.parameters(), lr=config["transition_model_lr"])
    criterion = nn.MSELoss()
    
    # Convert data to tensors
    states = torch.tensor([d[0] for d in data], dtype=torch.float32)
    actions = torch.tensor([d[1] for d in data], dtype=torch.float32)
    next_states = torch.tensor([d[2] for d in data], dtype=torch.float32)
    
    # Split into train and validation
    n_data = len(data)
    n_val = int(n_data * config["validation_split"])
    indices = list(range(n_data))
    random.shuffle(indices)
    train_idx, val_idx = indices[n_val:], indices[:n_val]
    
    train_dataset = torch.utils.data.TensorDataset(
        states[train_idx], actions[train_idx], next_states[train_idx]
    )
    val_dataset = torch.utils.data.TensorDataset(
        states[val_idx], actions[val_idx], next_states[val_idx]
    )
    
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=config["transition_model_batch_size"], shuffle=True
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=config["transition_model_batch_size"]
    )
    
    # Training loop
    for epoch in range(config["transition_model_epochs"]):
        model.train()
        train_loss = 0
        for batch_states, batch_actions, batch_next_states in train_loader:
            optimizer.zero_grad()
            pred_next_states = model(batch_states, batch_actions)
            loss = criterion(pred_next_states, batch_next_states)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch_states, batch_actions, batch_next_states in val_loader:
                pred_next_states = model(batch_states, batch_actions)
                val_loss += criterion(pred_next_states, batch_next_states).item()
        
        print(f"Epoch {epoch+1}/{config['transition_model_epochs']}, "
              f"Train Loss: {train_loss/len(train_loader):.4f}, "
              f"Val Loss: {val_loss/len(val_loader):.4f}")
    
    return model