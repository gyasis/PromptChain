"""
Test data generation utilities
"""
import uuid
import random
from datetime import datetime, timedelta
from faker import Faker
from typing import Dict, Any, List
from src.utils.logger import get_logger

logger = get_logger(__name__)

fake = Faker()

class TestDataManager:
    """Generates and manages test data for use cases."""

    def __init__(self):
        self.generated_data = {}
        self.cleanup_registry = []

    def generate_user_data(self, **overrides) -> Dict[str, Any]:
        """Generate realistic user registration data."""
        timestamp = int(datetime.now().timestamp())
        user_data = {
            'email': f'test.user.{timestamp}@example.com',
            'password': 'TestPass123!',
            'firstName': fake.first_name(),
            'lastName': fake.last_name(),
            'dateOfBirth': fake.date_of_birth(minimum_age=18, maximum_age=80).isoformat(),
            'phoneNumber': fake.phone_number(),
            **overrides
        }
        
        self.generated_data['last_user'] = user_data
        logger.debug(f"Generated user data: {user_data['email']}")
        return user_data

    def generate_profile_data(self, **overrides) -> Dict[str, Any]:
        """Generate user profile data."""
        profile_data = {
            'dateOfBirth': fake.date_of_birth(minimum_age=18, maximum_age=80).isoformat(),
            'phoneNumber': fake.phone_number(),
            'address': {
                'street': fake.street_address(),
                'city': fake.city(),
                'state': fake.state_abbr(),
                'zipCode': fake.zipcode(),
                'country': 'US'
            },
            'preferences': {
                'newsletter': True,
                'notifications': True,
                'language': 'en'
            },
            **overrides
        }
        
        self.generated_data['last_profile'] = profile_data
        return profile_data

    def generate_product_data(self, **overrides) -> Dict[str, Any]:
        """Generate product data."""
        product_data = {
            'name': f'Test Product {uuid.uuid4().hex[:8]}',
            'description': fake.text(max_nb_chars=200),
            'price': round(random.uniform(10.0, 500.0), 2),
            'sku': f'TEST-{uuid.uuid4().hex[:8].upper()}',
            'category': random.choice(['electronics', 'clothing', 'books', 'home']),
            'inventory': random.randint(10, 100),
            'tags': [fake.word() for _ in range(3)],
            **overrides
        }
        
        self.generated_data['last_product'] = product_data
        return product_data

    def generate_order_data(self, user_id: str = None, **overrides) -> Dict[str, Any]:
        """Generate order data."""
        order_data = {
            'userId': user_id or str(uuid.uuid4()),
            'items': [
                {
                    'productId': str(uuid.uuid4()),
                    'quantity': random.randint(1, 3),
                    'price': round(random.uniform(20.0, 100.0), 2)
                }
                for _ in range(random.randint(1, 4))
            ],
            'shippingAddress': {
                'street': fake.street_address(),
                'city': fake.city(),
                'state': fake.state_abbr(),
                'zipCode': fake.zipcode(),
                'country': 'US'
            },
            'paymentMethod': {
                'type': 'credit_card',
                'last4': fake.credit_card_number()[-4:],
                'token': f'tok_{uuid.uuid4().hex}'
            },
            'discountCode': None,
            **overrides
        }
        
        # Calculate total
        subtotal = sum(item['price'] * item['quantity'] for item in order_data['items'])
        order_data['subtotal'] = round(subtotal, 2)
        order_data['tax'] = round(subtotal * 0.08, 2)  # 8% tax
        order_data['shipping'] = 9.99
        order_data['total'] = round(order_data['subtotal'] + order_data['tax'] + order_data['shipping'], 2)
        
        self.generated_data['last_order'] = order_data
        return order_data

    def generate_file_data(self, file_type: str = 'image') -> Dict[str, Any]:
        """Generate file upload data."""
        file_types = {
            'image': {'extension': 'jpg', 'mime_type': 'image/jpeg'},
            'document': {'extension': 'pdf', 'mime_type': 'application/pdf'},
            'text': {'extension': 'txt', 'mime_type': 'text/plain'}
        }
        
        file_info = file_types.get(file_type, file_types['image'])
        
        return {
            'filename': f'test_file_{uuid.uuid4().hex[:8]}.{file_info["extension"]}',
            'mime_type': file_info['mime_type'],
            'size': random.randint(1024, 1024 * 1024),  # 1KB to 1MB
            'description': f'Test {file_type} file for validation'
        }

    def generate_custom_data(self, data_type: str = 'generic', **overrides) -> Dict[str, Any]:
        """Generate custom data based on type hint."""
        if data_type == 'auth':
            return {
                'username': fake.user_name(),
                'password': fake.password(length=12),
                'remember_me': random.choice([True, False])
            }
        elif data_type == 'contact':
            return {
                'name': fake.name(),
                'email': fake.email(),
                'subject': fake.sentence(),
                'message': fake.text()
            }
        else:
            return {
                'id': str(uuid.uuid4()),
                'timestamp': datetime.now().isoformat(),
                'random_value': fake.word(),
                **overrides
            }

    def register_for_cleanup(self, resource_type: str, resource_id: str):
        """Register a resource for cleanup."""
        self.cleanup_registry.append((resource_type, resource_id))
        logger.debug(f"Registered {resource_type} {resource_id} for cleanup")

    def get_cleanup_registry(self) -> List[tuple]:
        """Get list of resources to clean up."""
        return self.cleanup_registry.copy()

    def clear_registry(self):
        """Clear the cleanup registry."""
        self.cleanup_registry.clear()



