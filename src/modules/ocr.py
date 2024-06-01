import torch
import torch.nn as nn
import torchvision.models as models

class VGGFeatureExtractor(nn.Module):
    def __init__(self):
        super(VGGFeatureExtractor, self).__init__()
        vgg = models.vgg16(pretrained=True)
        self.features = vgg.features

    def forward(self, x):
        features = self.features(x)
        features = features.permute(0, 2, 3, 1)  # Rearrange to (batch, height, width, channels)
        features = features.view(features.size(0), -1, features.size(3))  # Flatten height and width
        return features

class Attention(nn.Module):
    def __init__(self, hidden_size):
        super(Attention, self).__init__()
        self.attention = nn.Linear(hidden_size, 1)

    def forward(self, hidden_states):
        # hidden_states: (batch_size, seq_length, hidden_size)
        attention_weights = torch.tanh(self.attention(hidden_states))
        attention_weights = torch.softmax(attention_weights, dim=1)
        return attention_weights

class GRUDecoder(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=1):
        super(GRUDecoder, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
        self.attention = Attention(hidden_size)
        self.fc = nn.Linear(hidden_size, output_size)
        
    def forward(self, features, hidden):
        # features: (batch_size, seq_length, input_size)
        gru_out, hidden = self.gru(features, hidden)
        attention_weights = self.attention(gru_out)
        context = attention_weights * gru_out
        context = context.sum(dim=1)  # (batch_size, hidden_size)
        
        output = self.fc(context)
        return output, hidden

    def init_hidden(self, batch_size):
        return torch.zeros(self.num_layers, batch_size, self.hidden_size).to(features.device)

class OCRModel(nn.Module):
    def __init__(self, feature_extractor, decoder, hidden_size):
        super(OCRModel, self).__init__()
        self.feature_extractor = feature_extractor
        self.decoder = decoder
        self.hidden_size = hidden_size
        
    def forward(self, x):
        features = self.feature_extractor(x)
        hidden = self.decoder.init_hidden(features.size(0))
        output, hidden = self.decoder(features, hidden)
        return output

# # Hyperparameters
# input_size = 512  # VGG feature size
# hidden_size = 256
# output_size = len(characters)  # Number of classes (characters)
# num_layers = 1

# # Instantiate the model
# vgg_extractor = VGGFeatureExtractor()
# decoder = GRUDecoder(input_size, hidden_size, output_size, num_layers)
# model = OCRModel(vgg_extractor, decoder, hidden_size)

# # Define loss and optimizer
# criterion = nn.CrossEntropyLoss()
# optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# # Training loop
# num_epochs = 10
# for epoch in range(num_epochs):
#     for batch in dataloader:
#         images = batch['image'].to(device)
#         labels = batch['label'].to(device)
        
#         # Forward pass
#         outputs = model(images)
        
#         # Compute loss
#         loss = criterion(outputs, labels)
        
#         # Backward pass and optimization
#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()
        
#     print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item()}')
